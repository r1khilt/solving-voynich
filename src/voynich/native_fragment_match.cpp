// Exact non-erasing 1/2-glyph template matches, with bounded ranked output.
// Different source rows MAY share the same unit. No injectivity constraint.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <queue>
#include <vector>

struct Match { int32_t gram, consumed; std::array<int32_t, 23> key; };
extern "C" int fragment_abi() { return 1; }
extern "C" int match_fragments(const int32_t* grams, const double* ranks,
    int32_t count, int32_t length, int32_t rows, int32_t glyphs,
    const int32_t* cipher, int32_t cipher_length, int32_t keep,
    uint64_t node_cap, Match* output, uint64_t* stats) {
  if (count < 1 || length < 1 || length > 12 || rows < 1 || rows > 23 ||
      glyphs < 1 || glyphs > 6 || cipher_length < 1 || keep < 1 || !node_cap) return 1;
  for (int i=0; i<count*length; ++i) if (grams[i]<0 || grams[i]>=rows) return 1;
  for (int i=0; i<cipher_length; ++i) if (cipher[i]<0 || cipher[i]>=glyphs) return 1;
  for (int i=0; i<2*count; ++i) if (!std::isfinite(ranks[i])) return 1;
  uint64_t nodes=0, matches=0;
  bool exceeded=false;
  // Better-than comparator leaves the WORST retained item at heap.top().
  auto better = [&](const Match& a, const Match& b, int arm) {
    double x=ranks[arm*count+a.gram], y=ranks[arm*count+b.gram];
    if (x!=y) return x>y;
    if (a.gram!=b.gram) return a.gram<b.gram;
    if (a.key!=b.key) return a.key<b.key;
    return a.consumed<b.consumed;
  };
  using Heap = std::priority_queue<Match, std::vector<Match>, std::function<bool(const Match&, const Match&)>>;
  Heap heaps[2] = {Heap([&](auto& a, auto& b){return better(a,b,0);}),
                   Heap([&](auto& a, auto& b){return better(a,b,1);})};
  std::array<int32_t,23> key;
  for (int gram=0; gram<count && !exceeded; ++gram) {
    key.fill(-1);
    auto visit = [&](auto&& self, int pos, int off) -> void {
      if (exceeded) return;
      if (nodes>=node_cap) { exceeded=true; return; }
      ++nodes;
      if (cipher_length-off < length-pos) return;
      if (pos==length) {
        ++matches;
        Match m{gram,off,key};
        for (int arm=0; arm<2; ++arm) {
          if (int(heaps[arm].size())<keep) heaps[arm].push(m);
          else if (better(m,heaps[arm].top(),arm)) { heaps[arm].pop(); heaps[arm].push(m); }
        }
        return;
      }
      int row=grams[gram*length+pos], unit=key[row];
      if (unit>=0) {
        int n=unit<glyphs ? 1 : 2;
        if (off+n>cipher_length) return;
        bool same = n==1 ? cipher[off]==unit :
          cipher[off]==(unit-glyphs)/glyphs && cipher[off+1]==(unit-glyphs)%glyphs;
        if (same) self(self,pos+1,off+n);
      } else {
        if (off<cipher_length) {
          key[row]=cipher[off]; self(self,pos+1,off+1);
        }
        if (off+1<cipher_length) {
          key[row]=glyphs+glyphs*cipher[off]+cipher[off+1]; self(self,pos+1,off+2);
        }
        key[row]=-1;
      }
    };
    visit(visit,0,0);
  }
  stats[0]=nodes; stats[1]=matches; stats[2]=0;
  if (exceeded) return 2; // Never publish a truncated scan as exact enumeration.
  std::vector<Match> retained;
  for (auto& heap : heaps) while (!heap.empty()) { retained.push_back(heap.top()); heap.pop(); }
  std::sort(retained.begin(),retained.end(),[](auto& a, auto& b){
    if (a.gram!=b.gram) return a.gram<b.gram;
    if (a.key!=b.key) return a.key<b.key;
    return a.consumed<b.consumed;
  });
  retained.erase(std::unique(retained.begin(),retained.end(),[](auto& a, auto& b){
    return a.gram==b.gram && a.key==b.key && a.consumed==b.consumed;
  }),retained.end());
  stats[2]=retained.size();
  std::copy(retained.begin(),retained.end(),output);
  return 0;
}
