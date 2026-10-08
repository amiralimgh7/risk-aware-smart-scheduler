# راهنمای فارسی تکمیلی

این فایل فقط راهنمای تکمیلی فارسی است. مستند اصلی پروژه در `README.md` و فایل‌های انگلیسی پوشه `docs/` قرار دارد.

## اجرای سریع روی ویندوز

از ریشه پروژه اجرا کنید:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --timeout 300 --retries 20 -r requirements_py313_windows.txt
$env:PYTHONPATH = (Get-Location).Path
```

تست‌ها:

```powershell
powershell -ExecutionPolicy Bypass -File .\docs\RUN_TESTS_WINDOWS.ps1
```

اجرای کامل:

```powershell
powershell -ExecutionPolicy Bypass -File .\docs\RUN_FULL_WINDOWS.ps1
```

اجرای فقط گزارش دفاع از خروجی‌های موجود:

```powershell
powershell -ExecutionPolicy Bypass -File .\docs\RUN_REPORT_ONLY_WINDOWS.ps1
```

هیچ کدام از اسکریپت‌های نهایی مسیر ثابت مربوط به یک سیستم خاص ندارند و مسیر پروژه را به صورت نسبی تشخیص می‌دهند.
