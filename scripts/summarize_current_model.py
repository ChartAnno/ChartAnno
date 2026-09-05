import argparse
import csv
import json
from pathlib import Path

def avg(values):
    nums = [float(v) for v in values if v not in ("", None)]
    return round(sum(nums) / len(nums), 4) if nums else ""

def resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p

def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the current model run into a per-sample combined CSV.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--output-dir", default="outputs/metric_summary")
    parser.add_argument("--run-dir", default="outputs")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_absolute():
        run_dir = repo_root / args.run_dir
    source_model = args.source_model
    
    # 1. Load High Level
    high_level_csv = run_dir / "api" / "semantic_design" / "high_level_scores_per_sample.csv"
    high_data = {}
    if high_level_csv.exists():
        with open(high_level_csv, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row["source_model"] != source_model: continue
                key = (row["chart_id"], row["category"], row["mode"], row["stage"])
                high_data[key] = row
            
    # 2. Load Chart Fidelity
    cf_csv = run_dir / "analysis" / "intermediates" / "chart_fidelity" / "chart_fidelity_all.csv"
    cf_data = {}
    if cf_csv.exists():
        with open(cf_csv, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row["model"] in ("LLM", "VLM"):
                    key = (row["chart_id"], row["category"], row["model"], row["layer"])
                    cf_data[key] = row["chart_fidelity"]
                    
    # 3. Load Anno Matching
    am_json = run_dir / "analysis" / "intermediates" / "annotation_matching" / "annotation_matching_results.json"
    am_data = {}
    if am_json.exists():
        with open(am_json, encoding='utf-8') as f:
            am = json.load(f)
            for m in ("LLM", "VLM"):
                for item in am.get("results", {}).get(m, {}).get("files", []):
                    category = item.get("category", "")
                    stage = item.get("stage", "")
                    pred_file = Path(item.get("pred_file", ""))
                    parts = pred_file.stem.split("_")
                    if len(parts) >= 2:
                        chart_id = f"{parts[0]}_{parts[1]}"
                    else:
                        chart_id = pred_file.stem
                    key = (chart_id, category, m, stage)
                    am_data[key] = item.get("overall", {}).get("jaccard", 0.0)
                            
    # 4. Load Color Matching
    cm_json = run_dir / "analysis" / "intermediates" / "color_matching" / "color_matching_results.json"
    cm_data = {}
    if cm_json.exists():
        with open(cm_json, encoding='utf-8') as f:
            cm = json.load(f)
            for item in cm.get("files", []):
                if item["model"] in ("LLM", "VLM"):
                    key = (item["chart_id"], item["category"], item["model"], item["stage"])
                    cm_data[key] = item["overall"]["f1"]
                    
    # 5. Get execution rate from test_images
    img_dir = run_dir / "test_images"
    
    # Merge
    out_rows = []
    
    # We iterate over the high_data keys because high level generated a row for every expected sample
    for key, h_row in high_data.items():
        cid, cat, mode, stage = key
        
        # Check if the code file exists to know if it was actually run/attempted
        code_dir_name = "code" if mode == "LLM" else "code+image"
        code_file_name = f"{cid}_code_{stage}.py" if mode == "LLM" else f"{cid}_code_image_{stage}.py"
        code_path = run_dir / "test_code" / cat / cid / code_dir_name / code_file_name
        
        if not code_path.exists():
            continue
            
        # Check image existence for execution_success_rate
        img_file_name = f"{cid}_code_{stage}.png" if mode == "LLM" else f"{cid}_code_image_{stage}.png"
        img_path = img_dir / cat / cid / code_dir_name / img_file_name
        img_exists = 1.0 if img_path.exists() else 0.0
        
        cf_val = cf_data.get(key, "")
        am_val = am_data.get(key, "")
        cm_val = cm_data.get(key, "")
        
        row = {
            "chart_id": cid,
            "category": cat,
            "mode": mode,
            "level": stage,
            "execution_success_rate": img_exists,
            "chart_fidelity": cf_val if cf_val != "" else 0.0,
            "annotation_matching": am_val if am_val != "" else 0.0,
            "color_matching": cm_val if cm_val != "" else 0.0,
            "semantic_faithfulness": h_row.get("semantic_faithfulness", ""),
            "semantic_clarity": h_row.get("semantic_clarity", ""),
            "visual_clarity": h_row.get("visual_clarity", ""),
            "annotation_organization_quality": h_row.get("annotation_organization_quality", ""),
            "attention_guidance": h_row.get("attention_guidance", ""),
            "structural_compliance": "",
            "semantic_consistency": h_row.get("semantic_consistency", ""),
            "design_effectiveness": h_row.get("design_effectiveness", "")
        }
        
        # In rule-based evaluation, intent doesn't have annotation matching or color matching.
        # Ensure we write empty strings for them to match the exact original format.
        if stage == "intent":
            row["annotation_matching"] = ""
            row["color_matching"] = ""
            row["structural_compliance"] = row["chart_fidelity"]
        else:
            sc_vals = []
            for k in ("chart_fidelity", "annotation_matching", "color_matching"):
                if row[k] not in ("", None): sc_vals.append(float(row[k]))
            if sc_vals:
                row["structural_compliance"] = round(sum(sc_vals) / len(sc_vals), 4)
                
        out_rows.append(row)
        
    overall_out = repo_root / "results" / "per_model_combined_csv" / f"{source_model}.csv"
    overall_out.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        "chart_id", "category", "mode", "level", 
        "execution_success_rate", "chart_fidelity", "annotation_matching", "color_matching",
        "semantic_faithfulness", "semantic_clarity", "visual_clarity",
        "annotation_organization_quality", "attention_guidance",
        "structural_compliance", "semantic_consistency", "design_effectiveness"
    ]
    
    with open(overall_out, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
        
    print(f"Saved: {overall_out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
