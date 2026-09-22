"""Verify and display the complete, exposed-development routing diagnosis."""

import gzip
import json
from pathlib import Path
import unicodedata

import numpy as np

from voynich.workspace.campaign import digest, write_json
from voynich.workspace.route_campaign import CONDITIONS, LAYERS, diagnostic, route_tasks, summaries


def normalized(text):
    value=unicodedata.normalize('NFKD',text.strip().strip(' .!\"\'`')).casefold()
    return ' '.join(''.join(c for c in value if not unicodedata.combining(c)).split())


def main():
    result,output=Path('results/ROUTE-0001'),Path('outputs/ROUTE-0001')
    report=json.loads((result/'results.json').read_text())
    manifest=json.loads((result/'inputs.json').read_text())
    assert all(digest(path)==expected for path,expected in manifest['source_sha256'].items())
    path=output/'observations.jsonl'
    assert digest(path)==report['observations_sha256']
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    tasks={r['id']:r for r in route_tasks()}
    keys={(r['id'],r['layer'],r['condition']) for r in rows}
    expected={(task,layer,condition) for task in tasks for layer in LAYERS for condition in CONDITIONS}
    assert len(keys)==len(rows)==1080 and keys==expected
    for row in rows:
        task=tasks[row['id']]
        for flag,text,answers in [('baseline_correct',row['baseline'],task['answers']),
                                  ('donor_correct',row['donor'],task['counterfactual_answers']),
                                  ('original_correct',row['answer'],task['answers']),
                                  ('counterfactual_correct',row['answer'],task['counterfactual_answers'])]:
            assert row[flag]==(normalized(text) in {normalized(a) for a in answers})
        if row['condition']=='identity':
            assert row['answer']==row['baseline']
        if row['condition']=='all_donor':
            assert row['first_token_matches_donor'] and row['full_donor_max_logit_error']<.002
    assert summaries(rows)==report['summary']
    saved=json.loads((result/'decision.json').read_text())
    assert all(saved[k]==v for k,v in diagnostic(report['summary']).items())
    archive=result/'observations.jsonl.gz'
    with archive.open('wb') as handle:
        with gzip.GzipFile(fileobj=handle,mode='wb',filename='',mtime=0) as compressed:
            compressed.write(path.read_bytes())
    assert gzip.decompress(archive.read_bytes())==path.read_bytes()
    complete_donor=[r for r in rows if r['condition']=='all_donor']
    audit={'passed':True,'rows':len(rows),'max_donor_first_logit_error':max(r['full_donor_max_logit_error'] for r in complete_donor),
           'all_donor_first_argmax_matches':sum(r['first_token_matches_donor'] for r in complete_donor),
           'all_donor_rows':len(complete_donor),
           'truncations':sum(r['truncated'] for r in rows),'archive_sha256':digest(archive),
           'observations_sha256':digest(path),'analysis_source_sha256':digest(__file__)}
    patterns={}
    lookup={(r['id'],r['layer'],r['condition']):r for r in rows}
    for layer in LAYERS:
        patterns[str(layer)]={}
        for intervention in ('donor','swap'):
            counts={}
            for task in tasks.values():
                if task['expected_effect']!='change':
                    continue
                group=[lookup[(task['id'],layer,scope+'_'+intervention)] for scope in ('last','earlier','all')]
                if not (group[0]['baseline_correct'] and group[0]['donor_correct']):
                    continue
                pattern=''.join(str(int(r['counterfactual_correct'])) for r in group)
                counts[pattern]=counts.get(pattern,0)+1
            patterns[str(layer)][intervention]=counts
    audit['descriptive_factorial_patterns']={'bit_order':['last','earlier','all'],'counts':patterns,
                                            'note':'Within-record intervention outcomes. 001 means only combined coverage succeeded; 011 means earlier and combined succeeded; 111 means either alone and combined succeeded. Descriptive, not a new pass criterion.'}
    audit['full_donor_first_agrees_but_complete_answer_fails']=[{'id':r['id'],'layer':r['layer']} for r in complete_donor
                                                              if r['expected_effect']=='change' and r['baseline_correct'] and r['donor_correct'] and not r['counterfactual_correct']]
    write_json(result/'audit.json',audit)
    plot(report['summary'],result)
    print(json.dumps(audit))


def plot(summary,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels=[(['last_donor','earlier_donor','all_donor'],['Final position','Earlier positions','Whole prompt'],'Broad donor-state replacement'),
            (['last_swap','earlier_swap','all_swap','all_random','all_raw'],['Final position','Earlier positions','Whole prompt','Matched random','Raw directions'],'Country-coordinate edits')]
    fig,axes=plt.subplots(1,2,figsize=(13,5.3),constrained_layout=True)
    for ax,(conditions,labels,title) in zip(axes,panels):
        x=np.arange(len(conditions))
        for index,(layer,color) in enumerate([('23','#2563eb'),('27','#059669')]):
            metrics=[summary[layer][condition]['counterfactual_both_clean_correct'] for condition in conditions]
            bars=ax.bar(x+(index-.5)*.35,[m['rate'] if m['rate'] is not None else 0 for m in metrics],width=.33,
                        color=color,label='Block '+str(int(layer)+1))
            for bar,metric in zip(bars,metrics):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.015,f"{metric['successes']}/{metric['count']}",ha='center',fontsize=8)
        ax.set(title=title,ylim=(0,1.15),ylabel='Complete donor answer | both clean answers correct')
        ax.set_xticks(x,labels,rotation=22,ha='right')
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    axes[0].legend(loc='upper left',fontsize=9)
    fig.suptitle('ROUTE-0001 | Does unchanged earlier context bypass a local edit?\nExploratory development reuse; shared facts and pairs, no independence claim',fontsize=13)
    fig.savefig(result/'position-coverage.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
