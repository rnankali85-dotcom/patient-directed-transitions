# راهنمای انتشار repository در GitHub

این فایل مخصوص اولین انتشار repository است.

## 1. ساخت حساب GitHub

در https://github.com حساب بسازید یا وارد حساب موجود شوید.

## 2. ساخت repository جدید

از GitHub گزینه `New repository` را انتخاب کنید.

پیشنهاد:

- Repository name: `patient-directed-transitions`
- Description: `Patient-specific directed transition features for episode-level outpatient healthcare utilization prediction`
- Visibility: `Public`
- گزینه Add a README را فعال نکنید؛ چون این package خودش README دارد.
- GitHub Actions یا License را در این مرحله لازم نیست اضافه کنید.

## 3. آماده‌سازی فایل‌ها

ZIP را extract کنید و وارد پوشه اصلی شوید.

مطمئن شوید این موارد وجود دارند:

- `README.md`
- `requirements.txt`
- `src/`
- `notebooks/`
- `data/synthetic_demo.csv`
- `figures/`
- `results/statistical_significance.csv`

هیچ فایل واقعی بیمار، CSV واقعی، hashed identifier واقعی، model مربوط به داده واقعی یا export از HIS نباید داخل repository باشد.

## 4. نصب Git

اگر Git نصب نیست، از سایت رسمی Git آن را نصب کنید.

بعد در Command Prompt یا PowerShell بررسی کنید:

```bash
git --version
```

## 5. اتصال پوشه به GitHub

در پوشه اصلی repository اجرا کنید:

```bash
git init
git add .
git status
```

در خروجی `git status` فایل‌هایی را که قرار است public شوند بررسی کنید.

## 6. اولین commit

```bash
git commit -m "Initial public release"
```

## 7. اتصال به repository GitHub

URL repository ساخته‌شده را از GitHub کپی کنید. سپس:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/patient-directed-transitions.git
git push -u origin main
```

به‌جای `YOUR_USERNAME` نام کاربری GitHub خودتان را قرار دهید.

## 8. بررسی repository در GitHub

بعد از push، در سایت GitHub بررسی کنید که:

- README درست نمایش داده می‌شود.
- `data/synthetic_demo.csv` وجود دارد.
- هیچ داده واقعی وجود ندارد.
- `models_demo/` وجود ندارد.
- `__pycache__/` وجود ندارد.
- فایل‌های SHAP چهارگانه وجود دارند.
- `statistical_significance.csv` وجود دارد.

## 9. اصلاح CITATION.cff

در `CITATION.cff` مقدار زیر را از placeholder به URL واقعی repository تغییر دهید:

```yaml
repository-code: "https://github.com/YOUR_USERNAME/patient-directed-transitions"
```

سپس:

```bash
git add CITATION.cff
git commit -m "Add permanent repository URL"
git push
```

## 10. اجرای تست بعد از clone

برای اطمینان از اینکه repository برای فرد دیگری هم قابل اجراست:

```bash
git clone https://github.com/YOUR_USERNAME/patient-directed-transitions.git
cd patient-directed-transitions
python -m venv .venv
```

فعال‌سازی محیط و نصب requirements را انجام دهید و سپس:

```bash
python src/run_demo.py
```

اگر پیام `Synthetic end-to-end demonstration completed.` مشاهده شد، مسیر public code قابل اجراست.

## 11. انتشار نسخه مشخص

بعد از اینکه repository نهایی شد، بهتر است یک release بسازید:

- GitHub → Releases → Draft a new release
- Tag: `v1.0.0`
- عنوان: `Initial reproducible release`
- توضیح کوتاه درباره synthetic demo و عدم انتشار داده‌های بیمار

این version/tag بعداً می‌تواند در مقاله به عنوان نسخه کد ذکر شود.

## 12. مهم برای مقاله

لینک GitHub را در manuscript فقط بعد از اینکه repository واقعاً public و بررسی‌شده شد قرار دهید.

تا قبل از آن از placeholder استفاده کنید و URL جعلی داخل مقاله قرار ندهید.
