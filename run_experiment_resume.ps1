# run_experiment_resume.ps1
#
# Chạy lại thí nghiệm so sánh LLM có/không ràng buộc cho đến khi đủ 78 case.
# Script tự bỏ qua các case đã hoàn thành trong results/llm_experiment/raw_cases.json
# nên có thể chạy bất kỳ lúc nào (kể cả khi hết hạn mức token hàng ngày của Groq —
# nó sẽ chờ 'rate limit' rồi chạy tiếp khi cửa sổ token trượt giải phóng quota).
#
# Cách dùng:  chạy từ thư mục gốc dự án
#   powershell -ExecutionPolicy Bypass -File run_experiment_resume.ps1

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

while ($true) {
    python -c "import json; r=json.load(open('results/llm_experiment/raw_cases.json',encoding='utf-8')); ok=sum(1 for x in r if x.get('constrained_text') and x.get('unconstrained_text')); print(f'HOÀN THÀNH {ok}/78'); exit(0 if ok>=78 else 1)"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Đã đủ 78 case. Chạy phân tích và xuất CSV..."
        python -m experiments.llm_unconstrained_experiment --only-analysis
        Write-Host "XONG. File: results/llm_experiment/cases_detail.csv và statistics_summary.csv"
        break
    }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Còn thieu case, tiep tuc generate (bo qua case da xong)..."
    python -m experiments.llm_unconstrained_experiment --model qwen/qwen3.6-27b --sleep 0.3
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Lan goi bi loi, cho 60s roi thu lai..."
        Start-Sleep -Seconds 60
    }
}