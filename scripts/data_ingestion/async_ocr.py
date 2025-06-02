import argparse
import asyncio
import glob
import json
import os
import sys
import pdb
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config_utils import load_config


def list_pending_pdfs(input_dir, output_dir):
    all_pdfs = sorted(input_dir.glob("*.pdf"))
    done = {p.stem for p in output_dir.glob("*.json")}
    return [p for p in all_pdfs if p.stem not in done]


async def ocr_pdf(session, pdf_path, sem, api_url, headers, max_retries, output_dir):
    filename = os.path.basename(pdf_path)
    
    for attempt in range(1, max_retries+1):
        async with sem:
            try:
                with open(pdf_path, "rb") as f:
                    data = aiohttp.FormData()
                    data.add_field("document", f, filename=filename, content_type="application/pdf")
                    data.add_field("model", "ocr")
                    async with session.post(api_url, headers=headers, data=data) as resp:
                        if resp.status != 200:
                            retry_after = resp.headers.get("Retry-After")
                            wait = float(retry_after) if retry_after else 2 ** (attempt-1)
                            print(f"[ERROR] {filename}: 재시도 {attempt}/{max_retries} in {wait:.1f}s")
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        result = await resp.json()

                output_dir.mkdir(exist_ok=True, parents=True)
                out_path = output_dir / f"{pdf_path.stem}.json"
                out_path.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
                print(f"[OK] {filename}")
                return True

            except Exception as e:
                if attempt == max_retries:
                    print(f"[FAIL] {filename}: {e}")
                    return False
                else:
                    backoff = 1 * attempt
                    print(f"[ERROR] {filename}: {e} → 재시도 {attempt}/{max_retries} in {backoff}s")
                    await asyncio.sleep(backoff)
    return False


async def main(args):
    config = load_config(args.config)
    root_path = Path(config["path"]["root_path"])
    cfg = config["async_ocr"]

    load_dotenv(PROJECT_ROOT/"config"/".env")

    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 UPSTAGE_API_KEY가 설정되지 않았습니다.")
    headers = {"Authorization": f"Bearer {api_key}"}

    api_url = cfg["api_url"]
    concurrent_worksers = cfg["concurrent_workers"]
    max_retries = cfg["max_retries"]

    input_dir = root_path/cfg["input_dir"]
    output_dir = root_path/cfg["output_dir"]
    failed_log = root_path/cfg["failed_log"]


    if failed_log.exists():
        failed_log.unlink()
    pendings = list_pending_pdfs(input_dir)
    print(f"총 PDF: {len(list(input_dir.glob('*.pdf')))}, 남은 OCR 대상: {len(pendings)}")

    sem  = asyncio.Semaphore(concurrent_worksers)
    failed_stems = []

    async with aiohttp.ClientSession() as session:
        tasks = [ocr_pdf(session, p, sem, api_url, headers, max_retries, output_dir) for p in pendings]
        for coro, pdf in zip(asyncio.as_completed(tasks), pendings):
            ok = await coro
            if not ok:
                failed_stems.append(pdf.stem)

    if failed_stems:
        with failed_log.open("w", encoding="utf-8") as fw:
            for s in failed_stems:
                fw.write(s + "\n")

    print("\n========== OCR 작업 완료 ==========")
    print(f"총 파일 수     : {len(pendings)}")
    print(f"성공한 파일 수 : {len(pendings) - len(failed_stems)}")
    print(f"실패한 파일 수 : {len(failed_stems)}")


if __name__ == "__main__":
    cur_dir = os.getcwd()

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=False, default="patent_invalidation_trial/config/config.json", help="Path of config.json")
    
    args = parser.parse_args()

    asyncio.run(main(args))

