# Kế hoạch hoàn thành Lab 22 (target: 100/100 + bonus)

Repo fork cá nhân, chưa chạy gì (chưa có venv, chưa có `models/`, `adapters/`, `data/pref`, `data/eval`).
Máy có GPU NVIDIA (driver OK) nhưng chưa cài torch. Quyết định môi trường chạy trước khi bắt đầu phần 1.

## 0. Chọn môi trường chạy — đã chọn: **Kaggle, 1-click "Save & Run All"**
- [x] Quyết định: chạy trên **Kaggle** bằng `colab/Lab22_DPO_T4.ipynb`, tier T4
- [x] Đã sửa `scripts/build_colab.py` + regenerate `colab/*.ipynb`: notebook tự nhận diện Kaggle
      (biến `KAGGLE_KERNEL_RUN_TYPE`) và ghi mọi output vào `/kaggle/working/lab22` — đây là thư mục
      Kaggle **thực sự lưu lại** khi commit, nên "Save & Run All" một phát là đủ, không cần copy tay
- [ ] Upload `colab/Lab22_DPO_T4.ipynb` lên Kaggle → New Notebook → Import Notebook/File
- [ ] Settings notebook: Accelerator = **GPU T4 x2**, **Internet = On** (cần pip install unsloth/trl/...),
      Persistence = mặc định là được (chỉ cần `/kaggle/working` được lưu, không cần bật gì thêm)
- [ ] Bấm **Save Version → Save & Run All (Commit)** (không dùng "Run All" ở chế độ Interactive — commit
      mới chạy nền độc lập, cho phép tắt máy/đóng tab mà vẫn tiếp tục chạy)
- [ ] Canh quota: Kaggle session batch tối đa ~12h, quota GPU ~30h/tuần — NB0–NB4 mất ~1,5–2h trên T4,
      đủ dư nếu chỉ chạy phần bắt buộc; nếu chạy `pipeline-full` (có bonus NB6 lm-eval, NB7 GRPO) nên
      ước lượng thêm thời gian trước khi rời máy
- [ ] Tối về mở tab "Output" của version đã commit, tải nguyên thư mục `lab22/` xuống máy — bên trong có
      `submission/screenshots/`, `data/eval/`, `adapters/dpo/*.json`, và notebook đã chạy xong (giữ output)
- [ ] Nếu commit bị lỗi giữa chừng (hết quota, lỗi pip...): xem log, sửa, chạy lại — vì ghi trực tiếp vào
      `/kaggle/working`, các file đã sinh ra ở cell trước lỗi vẫn còn trong Output của lần chạy đó

## 1. Phần bắt buộc — 100 điểm

### NB0 — `00_dpo_loss_from_scratch` (10 điểm)
- [ ] Điền `my_dpo_loss`, chạy qua hết `assert` (loss = log 2 khi mô hình == reference) — 6đ
- [ ] Trả lời câu hỏi "vì sao margin tăng được dù log-xác suất chosen giảm" ngay trong notebook — 4đ

### NB1 — `01_sft_mini` (8 điểm)
- [ ] `make nb0 sft` (hoặc chạy cell Colab) — xác nhận loss SFT giảm, ảnh `02-sft-loss.png` được tạo
- [ ] Xác nhận `models/sft-merged/` được lưu (đây là reference model cho DPO)

### NB2 — `02_preference_data` (12 điểm)
- [ ] `make data` — xác nhận assert "train/held-out không trùng câu hỏi" pass — 8đ
- [ ] Đọc kỹ 3 cặp mẫu in ra, ghi nhận ấn tượng (dùng cho REFLECTION §1)
- [ ] Xác nhận ảnh `02b-pref-length.png` + ghi lại tỉ lệ `chosen` dài hơn `rejected` — 4đ

### NB3 — `03_dpo_train` (24 điểm)
- [ ] `make dpo` — xác nhận adapter DPO train trên `models/sft-merged` (check config adapter) — 6đ
- [ ] Xác nhận `adapters/dpo/dpo_metrics.json` + ảnh `03-dpo-reward-curves.png` có **cả train và held-out**, tách riêng chosen/rejected — 10đ
- [ ] Đọc chẩn đoán tự động (INTENDED / LIKELIHOOD DISPLACEMENT / FAILURE / AMBIGUOUS), viết giải thích vào REFLECTION §3 (≥100 từ) — 8đ

### NB4 — `04_compare_and_eval` (16 điểm)
- [ ] `make eval` — xác nhận sinh câu trả lời cho 8 câu cố định + ≥50 câu held-out — 6đ
- [ ] Xác nhận `data/eval/judge_summary.json`: win rate + CI 95%, sanity accuracy, longer_answer_won_frac, length_matched_win_rate — 10đ
- [ ] Ảnh `04-side-by-side-table.png` tạo thành công

### Phản tư — `submission/REFLECTION.md` (20 điểm)
- [ ] §1–2: điền cấu hình + số liệu thật từ `dpo_metrics.json`
- [ ] §3 (≥100 từ): đọc đường reward, so train vs held-out, khớp/không khớp với chẩn đoán tự động
- [ ] §4: điền bảng win rate từ `judge_summary.json`, phân tích per_judge (thiên vị Skywork), chọn 2 ví dụ cụ thể (helpfulness + safety)
- [ ] §6 (≥150 từ): phân tích 1 quyết định quan trọng (β / lr / data / judge...)
- [ ] Kiểm tra §3+§6 tổng ≥ 150 từ

### Tái lập + Kiểm tra (10 điểm)
- [ ] Từ môi trường sạch: `make pipeline` chạy hết không lỗi (hoặc Colab "Chạy tất cả") — 5đ
- [ ] `make verify` kết thúc mã 0 — 5đ

### Nộp bài
- [ ] Copy ảnh/kết quả về `submission/screenshots/` và `data/eval/`, `adapters/dpo/*.json` nếu chạy Colab
- [ ] Commit NB0–NB4 đã chạy (giữ output), `REFLECTION.md`, screenshots
- [ ] Kiểm tra `.env`/API key KHÔNG bị commit (`.gitignore` đã chặn — double check `git status`)
- [ ] Push lên `https://github.com/awnpvng/K4-L3-Track3-Day22-DPO-ORPO-Alignment.git`, để public, nộp link vào LMS

## 2. Phần bonus (tối đa +20 điểm) — làm nếu còn thời gian/GPU

- [ ] NB3b — biến thể DPO/RPO/DPO-norm/LD-DPO/ORPO (+8): `make variants`, điền REFLECTION §8
- [ ] NB5 — GGUF Q4_K_M (+4): `make deploy`, so sánh HF vs GGUF trong `deploy_meta.json`
- [ ] NB6 — benchmark IFEval/GSM8K/Global-MMLU-vi (+6): `make bench`, điền REFLECTION §7 (≥150 từ, đọc stderr)
- [ ] NB7 — GRPO (+8): `make grpo`, điền REFLECTION §9 (accuracy trước/sau + nhiễu)
- [ ] β-sweep (+6): `make beta-sweep`, điền REFLECTION §5 (≥100 từ)
- [ ] Chấm chéo (+4): chạy NB4 với reward model + 1 giám khảo API khác họ, báo `cross_judge.agreement`
- [ ] Đẩy lên HF Hub (+3): push adapter + model card (base model, data, hyperparams, eval results)

## 3. Gợi ý thứ tự thực hiện thực tế

1. Chạy `make pipeline` end-to-end trước (NB0→NB4) để có baseline 100đ, review output mỗi bước.
2. Điền REFLECTION §1–4, §6 ngay khi có số liệu (đừng để cuối cùng mới viết).
3. Chạy `make verify`, sửa lỗi nếu có.
4. Nếu còn GPU time: làm bonus theo thứ tự điểm/công sức: β-sweep (6đ, rẻ) → NB3b (8đ) → NB6 (6đ) → NB7 (8đ) → NB5 (4đ) → chấm chéo (4đ) → HF Hub (3đ).
5. Commit + push, double-check public repo và không rò rỉ secrets.
