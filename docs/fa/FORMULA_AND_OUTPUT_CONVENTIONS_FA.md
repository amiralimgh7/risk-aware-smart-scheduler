# قراردادهای فرمولی و خروجی

## DMR

در کل پروژه، DMR فقط به معنی Deadline Miss Ratio است:

```text
deadline_miss_ratio = number_of_deadline_missed_tasks / number_of_total_tasks
```

وظایف حذف‌شده یا ارائه‌نشده به صورت جداگانه با `service_loss_ratio` گزارش می‌شوند.

## انرژی قبل و بعد از DVFS

مقایسه انرژی قبل و بعد از DVFS با همان کار تحقق‌یافته انجام می‌شود. حالت قبل، توان سطح بیشینه ولتاژ/بسامد را روی همان زمان‌بندی اعمال می‌کند و حالت بعد، سطح انتخاب‌شده توسط زمان‌بند را استفاده می‌کند.

## Mode Switch

تغییر حالت در هر اجرای DAG حداکثر یک‌بار رخ می‌دهد. سربار زمانی و انرژی آن در خروجی‌ها جدا ثبت می‌شود.

## Quantization

روش‌های قابل گزارش:

- FP32
- QAT INT8
- Weight-Only INT8 Per-Channel
- FP16 Mixed Precision

روش Dynamic INT8 Linear در خروجی نهایی گزارش نمی‌شود.
