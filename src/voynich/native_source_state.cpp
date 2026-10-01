// Markov sufficient-state history sums in consumed-glyph topological order.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using Key = std::array<int8_t,23>;
struct State {
  std::array<uint16_t,2> off{};
  std::array<uint32_t,2> ctx{};
  Key key{};
  uint64_t id=0;
  bool operator==(const State& b) const { return off==b.off && ctx==b.ctx && key==b.key && id==b.id; }
  bool operator<(const State& b) const {
    if (off!=b.off) return off<b.off;
    if (ctx!=b.ctx) return ctx<b.ctx;
    if (key!=b.key) return key<b.key;
    return id<b.id;
  }
};
struct KeyHash { size_t operator()(const Key& k) const {
  uint64_t h=1469598103934665603ULL;
  for (int8_t x:k) { h^=uint8_t(x); h*=1099511628211ULL; } return size_t(h);
}};
struct StateHash { size_t operator()(const State& s) const {
  uint64_t h=KeyHash{}(s.key);
  for (uint64_t x:{uint64_t(s.off[0]),uint64_t(s.off[1]),uint64_t(s.ctx[0]),uint64_t(s.ctx[1]),s.id}) {
    h^=x+0x9e3779b97f4a7c15ULL+(h<<6)+(h>>2);
  } return size_t(h);
}};
struct Value { double mass, best; };
struct Terminal { int32_t key[23]; double mass, best; };
struct Stats {
  uint64_t expanded,generated,merged,pruned,peak_active,peak_layer,terminals,returned,tables,trace_count;
  double found,returned_mass,lost_upper,evidence_upper;
  int32_t complete,stop;
};
static double add(double a,double b) {
  if (a==-INFINITY) return b; if (b==-INFINITY) return a;
  return std::max(a,b)+std::log1p(std::exp(-std::abs(a-b)));
}
struct IidGuide {
  struct Entry { std::vector<double> table; std::list<Key>::iterator position; };
  std::unordered_map<Key,Entry,KeyHash> cache;
  std::list<Key> lru;
  const int32_t* cipher; std::array<int32_t,2> lengths;
  int records,rows,glyphs,width; double rho,cont;
  uint64_t built=0,cap; size_t entries;
  std::vector<double> row;
  IidGuide(const int32_t* c,std::array<int32_t,2> n,int r,int a,int g,const double* p,
      double stop,uint64_t limit,size_t count):cipher(c),lengths(n),records(r),rows(a),glyphs(g),
      width(std::max(n[0],n[1])+2),rho(stop),cont(std::log1p(-stop)),cap(limit),entries(count) {
    for(int i=0;i<rows;++i) row.push_back((p[i]+1e-8)/(1+rows*1e-8));
  }
  double operator()(const State& s) {
    auto item=cache.find(s.key);
    if(item==cache.end()) {
      if(built>=cap) throw std::length_error("guide cap"); ++built;
      int units=glyphs+glyphs*glyphs;
      std::vector<double> weights(units,0.);
      double unknown=0.;
      for(int i=0;i<rows;++i) if(s.key[i]<0) unknown+=row[i];
      std::fill(weights.begin(),weights.end(),unknown/units);
      for(int i=0;i<rows;++i) if(s.key[i]>=0) weights[s.key[i]]+=row[i];
      for(double& w:weights) w=w ? std::log(w)+cont : -INFINITY;
      std::vector<double> table(records*width,-INFINITY);
      int start=0;
      for(int r=0;r<records;++r) {
        table[r*width+lengths[r]]=std::log(rho);
        for(int pos=lengths[r]-1;pos>=0;--pos) {
          double single=weights[cipher[start+pos]]+table[r*width+pos+1],dual=-INFINITY;
          if(pos+1<lengths[r]) {
            int code=glyphs+glyphs*cipher[start+pos]+cipher[start+pos+1];
            dual=weights[code]+table[r*width+pos+2];
          }
          table[r*width+pos]=add(single,dual);
        }
        start+=lengths[r];
      }
      lru.push_front(s.key);
      item=cache.emplace(s.key,Entry{std::move(table),lru.begin()}).first;
      if(cache.size()>entries) { cache.erase(lru.back()); lru.pop_back(); }
    } else lru.splice(lru.begin(),lru,item->second.position);
    double value=0.;
    for(int r=0;r<records;++r) if(s.off[r]<lengths[r]) value+=item->second.table[r*width+s.off[r]];
    return std::max(-2000.,value);
  }
};
extern "C" int source_state_abi() { return 1; }
extern "C" int source_state_search(const double* p,const uint32_t* goto_state,uint32_t states,
    int32_t rows,const int32_t* cipher,int32_t records,const int32_t* length_input,int32_t glyphs,
    double rho,uint64_t width,uint64_t max_expanded,uint64_t max_generated,uint64_t max_active,
    uint64_t max_terminals,int32_t merge,int32_t balanced,int32_t guided,uint64_t guide_cap,
    uint64_t guide_entries,uint64_t guide_prewidth,double max_seconds,Terminal* output,Stats* out,
    uint64_t* trace) {
  try {
    if(!states || rows<1 || rows>23 || records<1 || records>2 || glyphs<1 || glyphs>6 ||
       !(rho>0 && rho<1) || !width || !max_expanded || !max_generated || !max_active || !max_terminals ||
       !guide_cap || !guide_entries || !guide_prewidth || !(max_seconds>0)) return 1;
    std::array<int32_t,2> lengths{};
    for(int r=0;r<records;++r) { if(length_input[r]<1 || length_input[r]>4096) return 1; lengths[r]=length_input[r]; }
    int horizon=lengths[0]+lengths[1],start[2]={0,lengths[0]};
    const auto began=std::chrono::steady_clock::now();
    using Layer=std::unordered_map<State,Value,StateHash>;
    std::array<Layer,3> pending;
    State initial; initial.key.fill(-1);
    pending[0].emplace(initial,Value{0.,0.});
    std::unordered_map<Key,Value,KeyHash> terminal;
    Stats stats{}; stats.peak_active=1; stats.peak_layer=1;
    stats.found=stats.returned_mass=stats.lost_upper=-INFINITY;
    const double cont=std::log1p(-rho),stop=std::log(rho),penalty=std::log(glyphs+glyphs*glyphs);
    auto bound=[&](const State& s) {
      double value=0.;
      for(int r=0;r<records;++r) { int remaining=lengths[r]-s.off[r];
        if(remaining) { int low=(remaining+1)/2; value+=low*cont+std::log(-std::expm1((remaining-low+1)*cont)); }
      } return value;
    };
    IidGuide guide(cipher,lengths,records,rows,glyphs,p,rho,guide_cap,guide_entries);
    uint64_t serial=0,active=1;
    struct Candidate { State state; Value value; double rank; };
    auto better=[](const Candidate& a,const Candidate& b) {
      return a.rank!=b.rank ? a.rank>b.rank : a.state<b.state;
    };
    for(int layer=0;layer<=horizon;++layer) {
      Layer& here=pending[layer%3];
      if(here.empty()) continue;
      const uint64_t incoming=here.size();
      stats.peak_layer=std::max(stats.peak_layer,incoming);
      if(layer==horizon) {
        for(auto& item:here) {
          auto found=terminal.find(item.first.key);
          if(found==terminal.end()) terminal.emplace(item.first.key,item.second);
          else { found->second.mass=add(found->second.mass,item.second.mass); found->second.best=std::max(found->second.best,item.second.best); }
        }
        active-=incoming; here.clear(); continue;
      }
      std::vector<Candidate> candidates; candidates.reserve(incoming);
      for(auto& item:here) candidates.push_back({item.first,item.second,item.second.mass+bound(item.first)});
      active-=incoming;
      here.clear(); here.rehash(0);
      auto prune=[&](uint64_t keep) {
        if(candidates.size()>keep) {
          std::nth_element(candidates.begin(),candidates.begin()+keep,candidates.end(),better);
          for(size_t i=keep;i<candidates.size();++i) {
            stats.lost_upper=add(stats.lost_upper,candidates[i].value.mass+bound(candidates[i].state)); ++stats.pruned;
          }
          candidates.resize(keep);
        }
      };
      if(guided) { // Explicit extra beam stage; not an exact future likelihood.
        prune(guide_prewidth);
        for(auto& candidate:candidates) candidate.rank=candidate.value.mass+guide(candidate.state);
      }
      prune(width);
      std::sort(candidates.begin(),candidates.end(),better);
      double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
      int halt=stats.expanded+candidates.size()>max_expanded ? 1 :
        (stats.generated+2*rows*candidates.size()>max_generated ? 2 : (elapsed>=max_seconds ? 3 : 0));
      if(halt) {
        stats.stop=halt;
        for(auto& candidate:candidates) here.emplace(candidate.state,candidate.value);
        active+=candidates.size(); break;
      }
      for(auto& candidate:candidates) {
        State parent=candidate.state;
        int record=-1;
        for(int r=0;r<records;++r) if(parent.off[r]<lengths[r] && (record<0 ||
            (balanced && uint64_t(parent.off[r])*lengths[record]<uint64_t(parent.off[record])*lengths[r]))) record=r;
        int offset=parent.off[record]; uint32_t context=parent.ctx[record];
        ++stats.expanded;
        for(int row=0;row<rows;++row) {
          double probability=p[size_t(context)*rows+row];
          if(!probability) continue;
          int assigned=parent.key[row];
          for(int n=1;n<=2;++n) {
            if(offset+n>lengths[record]) continue;
            int code=n==1 ? cipher[start[record]+offset] :
              glyphs+glyphs*cipher[start[record]+offset]+cipher[start[record]+offset+1];
            if(assigned>=0 && assigned!=code) continue;
            State child=parent; child.off[record]+=n;
            child.ctx[record]=goto_state[size_t(context)*rows+row];
            double edge=cont+std::log(probability);
            if(assigned<0) { child.key[row]=int8_t(code); edge-=penalty; }
            if(child.off[record]==lengths[record]) { child.ctx[record]=0; edge+=stop; }
            ++stats.generated; ++serial; child.id=merge ? 0 : serial;
            int rank=child.off[0]+child.off[1];
            Layer& target=pending[rank%3];
            Value value{candidate.value.mass+edge,candidate.value.best+edge};
            auto found=target.find(child);
            if(found==target.end()) { target.emplace(child,value); ++active; }
            else { found->second.mass=add(found->second.mass,value.mass); found->second.best=std::max(found->second.best,value.best); ++stats.merged; }
            stats.peak_active=std::max(stats.peak_active,active);
            if(active>max_active) return 2;
          }
        }
      }
      uint64_t* t=trace+8*stats.trace_count++;
      t[0]=layer;t[1]=incoming;t[2]=candidates.size();t[3]=stats.expanded;t[4]=stats.generated;
      t[5]=stats.merged;t[6]=stats.pruned;t[7]=active;
    }
    double unresolved=-INFINITY;
    for(auto& layer:pending) for(auto& item:layer) unresolved=add(unresolved,item.second.mass+bound(item.first));
    stats.lost_upper=add(stats.lost_upper,unresolved);
    std::vector<std::pair<Key,Value>> final;
    for(auto& item:terminal) { stats.found=add(stats.found,item.second.mass); final.push_back(item); }
    std::sort(final.begin(),final.end(),[](auto& a,auto& b) {
      return a.second.mass!=b.second.mass ? a.second.mass>b.second.mass : a.first<b.first;
    });
    stats.terminals=final.size(); stats.returned=std::min(uint64_t(final.size()),max_terminals);
    for(uint64_t i=0;i<stats.returned;++i) {
      for(int r=0;r<23;++r) output[i].key[r]=final[i].first[r];
      output[i].mass=final[i].second.mass; output[i].best=final[i].second.best;
      stats.returned_mass=add(stats.returned_mass,final[i].second.mass);
    }
    stats.evidence_upper=add(stats.found,stats.lost_upper);
    stats.complete=stats.pruned==0 && active==0;
    stats.tables=guide.built; *out=stats; return 0;
  } catch(const std::length_error&) { return 3; }
    catch(const std::exception&) { return 4; }
}
