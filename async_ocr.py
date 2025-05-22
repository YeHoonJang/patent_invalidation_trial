# 비동기 OCR 스크립트: 에러·성공 카운트 및 최종 리포트
import os
import glob
import asyncio
import json
import aiohttp
from dotenv import load_dotenv
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# 환경 설정
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv("/home/yehoon/workspace/patent_invalidation_trial/config/.env")
API_KEY = os.getenv("UPSTAGE_API_KEY")
if not API_KEY:
    raise RuntimeError("환경변수 UPSTAGE_API_KEY가 설정되지 않았습니다.")

API_URL = "https://api.upstage.ai/v1/document-digitization"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
INPUT_DIR = Path("/home/yehoon/workspace/patent_invalidation_trial/data/ptab_21c_v2/cover")
OUTPUT_DIR = Path("/home/yehoon/workspace/patent_invalidation_trial/data/ocr_results")
FAILED_LOG = Path("/home/yehoon/workspace/patent_invalidation_trial/log")
CONCURRENT_WORKERS = 3
MAX_RETRIES = 5
# ──────────────────────────────────────────────────────────────────────────────


def list_pending_pdfs():
    all_pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    done = {p.stem for p in OUTPUT_DIR.glob("*.json")}
    return [p for p in all_pdfs if p.stem not in done]


async def ocr_pdf(session, pdf_path, sem):
    filename = os.path.basename(pdf_path)
    
    for attempt in range(1, MAX_RETRIES+1):
        async with sem:
            try:
                with open(pdf_path, "rb") as f:
                    data = aiohttp.FormData()
                    data.add_field("document", f, filename=filename, content_type="application/pdf")
                    data.add_field("model", "ocr")
                    async with session.post(API_URL, headers=HEADERS, data=data) as resp:
                        if resp.status != 200:
                            retry_after = resp.headers.get("Retry-After")
                            wait = float(retry_after) if retry_after else 2 ** (attempt-1)
                            print(f"[ERROR] {filename}: 재시도 {attempt}/{MAX_RETRIES} in {wait:.1f}s")
                            await asyncio.sleep(wait)
                            continue
                        resp.raise_for_status()
                        result = await resp.json()

                OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
                out_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
                out_path.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
                print(f"[OK] {filename}")
                return True

            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"[FAIL] {filename}: {e}")
                    return False
                else:
                    backoff = 1 * attempt
                    print(f"[ERROR] {filename}: {e} → 재시도 {attempt}/{MAX_RETRIES} in {backoff}s")
                    await asyncio.sleep(backoff)
    return False


async def main():
    if FAILED_LOG.exists():
        FAILED_LOG.unlink()
    pendings = list_pending_pdfs()
    print(f"총 PDF: {len(list(INPUT_DIR.glob('*.pdf')))}, 남은 OCR 대상: {len(pendings)}")

    sem  = asyncio.Semaphore(CONCURRENT_WORKERS)
    failed_stems = []

    async with aiohttp.ClientSession() as session:
        tasks = [ocr_pdf(session, p, sem) for p in pendings]
        for coro, pdf in zip(asyncio.as_completed(tasks), pendings):
            ok = await coro
            if not ok:
                failed_stems.append(pdf.stem)

    if failed_stems:
        with FAILED_LOG.open("w", encoding="utf-8") as fw:
            for s in failed_stems:
                fw.write(s + "\n")

    print("\n========== OCR 작업 완료 ==========")
    print(f"총 파일 수     : {len(pendings)}")
    print(f"성공한 파일 수 : {len(pendings) - len(failed_stems)}")
    print(f"실패한 파일 수 : {len(failed_stems)}")

if __name__ == "__main__":
    asyncio.run(main())

