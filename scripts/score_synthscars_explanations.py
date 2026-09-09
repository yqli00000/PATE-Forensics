from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.synthscars_protocol import load_records, normalize_text, reference_explanation, rouge_l_f1


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--data-root",type=Path,required=True); ap.add_argument("--annotation-json",type=Path,default=None); ap.add_argument("--predictions-jsonl",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--model",default="sentence-transformers/paraphrase-MiniLM-L6-v2")
    ap.add_argument("--device",default="cuda"); ap.add_argument("--limit",type=int,default=None,
                    help="Score the first N records in official annotation order and ignore predictions outside that subset.")
    args=ap.parse_args()
    if args.limit is not None and args.limit <= 0:
        ap.error("limit must be positive")
    prediction_rows=[json.loads(line) for line in args.predictions_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    preds={str(x["uid"]):x for x in prediction_rows}
    if len(preds) != len(prediction_rows):
        ap.error("duplicate prediction UIDs")
    annotation_json=args.annotation_json or args.data_root/"test/annotations/test.json"
    all_records=load_records(annotation_json)
    if args.limit is not None:
        selected_records=all_records[:args.limit]
    else:
        selected_records=[record for record in all_records if record.uid in preds]
        unknown=sorted(set(preds)-{record.uid for record in selected_records})
        if unknown:
            raise SystemExit(f"prediction UIDs absent from annotation: {unknown[:20]}")
    missing=[record.uid for record in selected_records if record.uid not in preds]
    if missing:
        raise SystemExit(f"missing predictions for selected annotation records: {missing[:20]}")
    ordered=[(record.uid,normalize_text(preds[record.uid]["explanation"]),reference_explanation(record))
             for record in selected_records]
    if not ordered:
        ap.error("no predictions selected for scoring")
    model=SentenceTransformer(args.model, device=args.device)
    p=model.encode([x[1] for x in ordered],convert_to_numpy=True,normalize_embeddings=True)
    r=model.encode([x[2] for x in ordered],convert_to_numpy=True,normalize_embeddings=True)
    css=np.sum(p*r,axis=1); rows=[]
    for (uid,pred,ref),score in zip(ordered,css): rows.append({"uid":uid,"prediction":pred,"reference":ref,"rouge_l":rouge_l_f1(pred,ref),"css":float(score)})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/"explanation_per_sample.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8")
    summary={"num_samples":len(rows),"reference":"top-level caption","preprocessing":"collapse all whitespace and strip, identically for prediction/GT",
             "rouge_l_definition":"token-level LCS F1","css_model":args.model,"rouge_l_0_100":100*float(np.mean([x['rouge_l'] for x in rows])),"css_0_100":100*float(css.mean())}
    (args.output_dir/"explanation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
