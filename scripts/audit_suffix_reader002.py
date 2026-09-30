"""Reverse inference for the depth-mass model; separate from its table builder."""
import math


def probability(raw, history, char):
    counts = raw['counts']
    root = counts['']
    result = (root.get(char,0)+.5)/(sum(root.values())+.5*len(raw['alphabet']))
    for depth in range(1, min(len(history),len(raw['masses']))+1):
        row=counts.get(history[-depth:])
        if row is not None:
            mass=raw['masses'][depth-1]
            result=(row.get(char,0)+mass*result)/(sum(row.values())+mass)
    return result


def infer(raw, units, observed, rho, max_nodes=500_000):
    alphabet, counts, order = raw['alphabet'], raw['counts'], len(raw['masses'])
    def following(history,char):
        history=(history+char)[-order:] if order else ''
        while history not in counts:
            history=history[1:]
        return history
    n=len(observed)
    reachable=[set() for _ in range(n+1)]
    reachable[0].add('')
    matches=[[(char,i+len(unit)) for char,unit in zip(alphabet,units,strict=True)
              if observed.startswith(unit,i)] for i in range(n)]
    nodes=1
    for i in range(n):
        for context in reachable[i]:
            for char,end in matches[i]:
                after=following(context,char)
                if after not in reachable[end]:
                    nodes+=1
                    if nodes>max_nodes:
                        raise RuntimeError('Reference node cap')
                    reachable[end].add(after)
    total={(n,h):math.log(rho) for h in reachable[n]}
    best=dict(total)
    weights={}
    for i in range(n-1,-1,-1):
        for context in reachable[i]:
            terms,maxima=[],[]
            for char,end in matches[i]:
                destination=end,following(context,char)
                if destination not in total:
                    continue
                if (context,char) not in weights:
                    weights[context,char]=math.log1p(-rho)+math.log(probability(raw,context,char))
                weight=weights[context,char]
                terms.append(weight+total[destination])
                maxima.append(weight+best[destination])
            if terms:
                high=max(terms)
                total[i,context]=high+math.log(math.fsum(math.exp(v-high) for v in terms))
                best[i,context]=max(maxima)
    return total.get((0,''),-math.inf),best.get((0,''),-math.inf),nodes


def reading(raw, units, observed, text, rho):
    if text is None:
        return -math.inf
    mapping=dict(zip(raw['alphabet'],units,strict=True))
    if ''.join(mapping[c] for c in text)!=observed:
        raise ValueError('Reading does not re-encode')
    return math.fsum([math.log(rho), *[math.log1p(-rho)+math.log(probability(raw,text[:i],c))
                                     for i,c in enumerate(text)]])
