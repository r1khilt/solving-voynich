import itertools
import json
import math

import numpy as np
import pytest

from scripts import audit_borg_common001 as audit
from voynich import borg_common as model


def test_independent_source_every_tiny_context_and_literal_prefix_scoring():
    alphabet = "abc"
    text = "abacabaacbbcabcacbaabccbbabcaa"
    tables = model.train_source(text,alphabet)
    source = audit.independent_source(text,alphabet)
    for n in range(4):
        for digits in itertools.product(range(3),repeat=n):
            context = "".join(alphabet[i] for i in digits)
            for target in range(3):
                code = 0
                for value in (*digits,target):
                    code = 3*code+value
                assert source(context,alphabet[target]) == pytest.approx(tables[n][code],abs=1e-14)
            assert math.fsum(math.exp(source(context,c)) for c in alphabet) == pytest.approx(1.,abs=1e-14)
    records = ["a","bc","abca","ccbbaab","ca"]
    objective = model.Objective(model.encode_records(records,alphabet),tables,source_size=3,observed_size=3)
    keys = np.array(list(itertools.permutations(range(3))))
    for key,score in zip(keys,objective.scores(keys),strict=True):
        decoded = audit.literal_decode(records,key.tolist(),alphabet,alphabet)
        assert score == pytest.approx(audit.literal_score(decoded,source),abs=1e-13)


def test_independent_metrics_and_strict_integer_key(monkeypatch):
    monkeypatch.setattr(audit.run,"COMMON","ABC")
    records = ["ABCA","BC"]
    key = [2,0,1]
    predictions = audit.literal_decode(records,key,"ABC","abc")
    truth = {"plain_transfer":predictions,"key":key}
    assert audit.independent_metrics(records,predictions,truth,key) == {
        "errors":0,"characters":6,"used_rows":3,"correct_used_rows":3,"pass":True}
    wrong = dict(truth,plain_transfer=["caab","ab"])
    assert not audit.independent_metrics(records,predictions,wrong,key)["pass"]
    with pytest.raises(ValueError):
        audit.literal_decode(records,[2.,0.,1.],"ABC","abc")
    with pytest.raises(ValueError):
        audit.literal_decode(records,[True,0,2],"ABC","abc")
    with pytest.raises(ValueError):
        audit.close_number(float("nan"),0)
    with pytest.raises(ValueError):
        audit.close_number(2.,1.)


@pytest.mark.parametrize("corruption",(None,"score","metric","source_hash","empty_fit"))
def test_complete_temporary_audit_artifact_accounting_and_exclusive_outputs(tmp_path,monkeypatch,corruption):
    run = audit.run
    root, out, bulk = tmp_path,tmp_path/"results"/run.EXP,tmp_path/"outputs"/run.EXP
    monkeypatch.setattr(run,"ROOT",root)
    monkeypatch.setattr(run,"OUT",out)
    monkeypatch.setattr(run,"BULK",bulk)
    monkeypatch.setattr(run,"PREP",out/"prepare.json")
    monkeypatch.setattr(run,"RAW","raw.txt")
    monkeypatch.setattr(run,"PATHS",("fixture.txt",))
    (root/"fixture.txt").write_text("Artificial integration fixture, no original fit.\n")
    crypt = run.COMMON
    (root/"raw.txt").write_bytes(crypt.encode())
    piece = {"start_byte":0,"end_byte":len(crypt),"raw_sha256":run.digest(crypt.encode()),"symbols":crypt}
    historical = {"symbols":crypt,"pieces":[piece]}
    controls = {name:{"fit":[crypt],"transfer":[crypt[::-1]],"search_seed":seed,
                       "kind":"positive" if name.startswith("control") else "paired_row_shuffle"}
                for name,seed in (("control-0",96721),("control-1",96729),("shuffle-0",96821),("shuffle-1",96829))}
    inputs = {"experiment":run.EXP,"historical":{"fit":[historical],"transfer":[historical]},"controls":controls}
    key = list(range(len(run.LATIN)))
    gold = {name:{"key":key,"plain_transfer":audit.literal_decode(row["transfer"],key)}
            for name,row in controls.items() if name.startswith("control")}
    sources = {language:{"entry":{"fixture":language},"text":text}
               for language,text in (("latin",run.LATIN*4),("english",run.LATIN[::-1]*4))}
    metadata = {"source_files":{language:row["entry"] for language,row in sources.items()}}
    input_artifact = run.save_new(bulk/"inputs.json.gz",inputs,compressed=True)
    gold_artifact = run.save_new(bulk/"gold.json.gz",gold,compressed=True)
    run.save_new(run.PREP,{"inputs":input_artifact,"control_gold":gold_artifact,**metadata})
    monkeypatch.setattr(run,"prepare_payload",lambda:(inputs,gold,metadata))
    monkeypatch.setattr(run,"source_inputs",lambda:sources)
    paths = (*run.PATHS,str(run.PREP.relative_to(root)))
    hashes = {path:run.digest((root/path).read_bytes()) for path in paths}
    monkeypatch.setattr("subprocess.check_output",lambda command,**_: (root/command[-1].split(":",1)[1]).read_bytes())
    run.save_new(out/"started.json",{"experiment":run.EXP,"freeze":"fixture","started_unix":1,"source_hashes":hashes})
    models = {language:run.train_source(row["text"]) for language,row in sources.items()}
    fits = {}
    for name,(records,language,seed) in audit.expected_fit_inputs(inputs).items():
        obj = model.Objective(model.encode_records(records,run.COMMON),models[language])
        score = float(obj.scores(np.array([key]))[0])
        endpoints = [{"restart":i,"key":key,"score":score,"sweeps":1,"changes":0} for i in range(run.RESTARTS)]
        fits[name] = {"key":key,"score":score,"language":language,"seed":seed,"restarts":run.RESTARTS,
                      "max_sweeps":run.SWEEPS,"endpoints":endpoints,"fit_characters":sum(map(len,records)),
                      "fit_records":len(records),"input_sha256":run.digest(json.dumps(records).encode())}
    seal = {"experiment":run.EXP,"freeze":"fixture","source_hashes":hashes,"fits":fits,
            "preparation_sha256":run.digest(run.PREP.read_bytes()),"historical_answer_opened":False,"seal_unix":2}
    seal_artifact = run.save_new(out/"sealed_predictions.json",seal)
    transfers,metrics = {},{}
    for name in (*controls,"borg-latin","borg-english"):
        language = "english" if name == "borg-english" else "latin"
        records = controls[name]["transfer"] if name in controls else [crypt]
        decoded = audit.literal_decode(records,key)
        obj = model.Objective(model.encode_records(records,run.COMMON),models[language])
        transfers[name] = {"decoded":decoded,"score":float(obj.scores(np.array([key]))[0]),"characters":sum(map(len,records))}
        if name in gold:
            metrics[name] = audit.independent_metrics(records,decoded,gold[name],key)
    private = run.save_new(bulk/"transfers.json.gz",transfers,compressed=True)
    run.save_new(out/"result.json",{"experiment":run.EXP,"freeze":"fixture","seal":seal_artifact,
        "private_transfers":private,"control_metrics":metrics,"control_competence":"PASS",
        "historical_answer_opened":False,"historical_recovery":"UNASSESSED_NO_EXTERNAL_ANSWER",
        "transfer_scores":{name:{k:v for k,v in row.items() if k != "decoded"} for name,row in transfers.items()},
        "cpu_seconds":.1,"wall_seconds":.1,"peak_rss_bytes":1000,"paid_cost_usd":0})
    if corruption:
        if corruption in ("score","source_hash","empty_fit"):
            changed = json.loads((out/"sealed_predictions.json").read_text())
            if corruption == "score":
                changed["fits"]["borg-latin"]["endpoints"][0]["score"] += 1
            elif corruption == "source_hash":
                changed["source_hashes"]["fixture.txt"] = "0"*64
            else:
                changed["fits"] = {}
            (out/"sealed_predictions.json").write_text(json.dumps(changed))
        else:
            changed = json.loads((out/"result.json").read_text())
            changed["control_metrics"]["control-0"]["errors"] = 1
            (out/"result.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            audit.audit("fixture")
        assert (out/"audit-failure.json").exists() and not (out/"audit.json").exists()
        return
    assert audit.audit("fixture") == "PASS_literal_scores_selection_and_accounting"
    receipt = json.loads((out/"audit.json").read_text())
    assert receipt["restart_endpoint_literal_score_checks"] == 84 and receipt["transfer_record_sets"] == 6
    assert receipt["control_metrics"] == metrics and receipt["retained_raw_piece_checks"] == 2
    with pytest.raises(FileExistsError):
        audit.audit("fixture")
