# CatVTON RunPod Serverless Worker

یک worker مستقل و production-oriented برای Virtual Try-On با [مخزن رسمی CatVTON](https://github.com/Zheng-Chong/CatVTON). سرویس دو تصویر base64 می‌گیرد، mask لباس را با DensePose/SCHP می‌سازد، CatVTON را روی GPU اجرا می‌کند و JPEG خروجی را به‌صورت base64 برمی‌گرداند. این repository هیچ کدی از Telegram Bot ندارد.

## معماری و رفتار

- مدل فقط یک‌بار در startup پردازش worker و پیش از `runpod.serverless.start` بارگذاری می‌شود.
- نبود CUDA باعث fail-fast شدن startup با پیام واضح می‌شود.
- pipeline، AutoMasker و وزن‌ها بین jobها reuse می‌شوند؛ inference با `torch.float16` و `torch.inference_mode()` اجرا می‌شود.
- برای ایمنی GPU، درخواست‌های هم‌زمان داخل یک worker توسط lock سری می‌شوند. scale-out را با تعداد workerهای RunPod انجام دهید.
- تصویر شخص با حفظ نسبت تصویر resize و center-crop، و تصویر لباس با حفظ نسبت resize و padding می‌شود. رزولوشن پیش‌فرض `768x1024` است.
- CatVTON کلاسیک prompt متنی را condition نمی‌کند؛ فیلد `prompt` برای سازگاری API پذیرفته و اعتبارسنجی می‌شود اما در inference مصرف نمی‌شود.
- تصویر یا base64 در log نوشته نمی‌شود و این implementation فایل موقت ایجاد نمی‌کند؛ bufferهای حافظه و PIL imageها در پایان هر job آزاد می‌شوند.

## فایل‌ها

```text
.
├── handler.py              # RunPod entrypoint و پاسخ API
├── inference.py            # interface ساده generate_tryon
├── model_loader.py         # singleton startup model loader
├── image_utils.py          # decode/verify/EXIF/resize/JPEG encode
├── schemas.py              # validation و خطاهای public
├── requirements.txt
├── Dockerfile
├── tests/test_handler.py
└── scripts/local_test.py
```

## پیش‌نیازها

- برای اجرای واقعی: Linux، Docker، NVIDIA GPU با حداقل حدود 8 GB VRAM، driver سازگار با CUDA 12.1 و NVIDIA Container Toolkit
- برای build/push: Docker و یک حساب Docker Hub
- برای deploy: حساب RunPod، موجودی حساب، و دسترسی به Serverless
- برای تست واحد فقط Python 3.9+ لازم است و GPU یا model weights لازم نیست.

توجه: اجرای native روی Windows برای stack رسمی Detectron2/CatVTON توصیه نمی‌شود؛ Docker روی Linux/RunPod مسیر پشتیبانی‌شده این پروژه است.

## API

ورودی handler همان envelope استاندارد RunPod است:

```json
{
  "input": {
    "person_image_base64": "...",
    "garment_image_base64": "...",
    "garment_category": "upper_body",
    "prompt": "",
    "seed": 42,
    "steps": 30
  }
}
```

`garment_category` یکی از `upper_body`، `lower_body` یا `dress` است. `seed` باید integer بین `0` و `4294967295` و `steps` بین `1` و `100` باشد. هر تصویر باید JPEG، PNG یا WEBP معتبر و حداکثر 10 MiB بعد از decode باشد. data URL نیز پذیرفته می‌شود.

پاسخ موفق:

```json
{
  "status": "completed",
  "output": {
    "image_base64": "...",
    "mime_type": "image/jpeg",
    "width": 768,
    "height": 1024,
    "seed": 42
  },
  "metrics": {
    "inference_seconds": 12.345,
    "total_seconds": 12.401
  }
}
```

پاسخ validation یا runtime error، بدون traceback برای client:

```json
{
  "status": "failed",
  "error": {
    "code": "INVALID_IMAGE",
    "message": "person_image_base64 is not valid base64."
  }
}
```

کدهای رایج: `INVALID_INPUT`، `MISSING_PERSON_IMAGE`، `MISSING_GARMENT_IMAGE`، `INVALID_IMAGE`، `UNSUPPORTED_IMAGE_FORMAT`، `IMAGE_TOO_LARGE`، `INVALID_CATEGORY`، `INVALID_SEED`، `INVALID_STEPS` و `INFERENCE_ERROR`.

## تست و اجرای local

تست‌های واحد (بدون GPU):

```bash
python -m venv .venv
source .venv/bin/activate
pip install Pillow pytest
python -m compileall .
pytest -q
```

اجرای inference واقعی باید در محیطی انجام شود که source رسمی CatVTON، dependencyها، CUDA و وزن‌ها موجود باشند. ساده‌ترین روش، اجرای shell در container ساخته‌شده است:

```bash
docker run --rm --gpus all \
  -v "$PWD/samples:/app/samples:ro" \
  -v "$PWD/output:/app/output" \
  -v catvton-hf-cache:/models/huggingface \
  --entrypoint python catvton-runpod-worker:latest \
  scripts/local_test.py \
  --person samples/person.jpg \
  --garment samples/garment.jpg \
  --category upper_body \
  --output output/result.jpg
```

برای اجرای مستقیم روی یک Linux GPU host، ابتدا repository رسمی را در `/opt/CatVTON` clone کنید و Detectron2 و DensePose را مطابق `INSTALL.md` رسمی CatVTON از repository رسمی Detectron2 نصب کنید؛ سپس `python -m pip install -r requirements.txt` و همان دستور `python scripts/local_test.py ...` را اجرا کنید. مسیر source با `CATVTON_SOURCE_DIR` قابل تغییر است.

## Model download و cache

Dockerfile کد رسمی را از `Zheng-Chong/CatVTON` clone می‌کند. برای build قابل تکرار، `CATVTON_REF` را به commit SHA بررسی‌شده خود pin کنید:

```bash
docker build \
  --build-arg CATVTON_REF=<VERIFIED_COMMIT_SHA> \
  -t YOUR_DOCKERHUB_USER/catvton-runpod-worker:1.0.0 .
```

به‌طور پیش‌فرض وزن‌های `zhengchong/CatVTON`، مدل inpainting پایه و VAE در اولین startup توسط Hugging Face دانلود و زیر `HF_HOME=/models/huggingface` cache می‌شوند. دو روش deployment وجود دارد:

1. **image کوچک‌تر، cold start طولانی‌تر:** build پیش‌فرض و استفاده از RunPod cached model/network volume.
2. **image بزرگ‌تر، startup قابل‌پیش‌بینی‌تر:** وزن‌ها را در build bake کنید:

```bash
docker build \
  --build-arg BAKE_MODEL_WEIGHTS=true \
  --build-arg CATVTON_REF=<VERIFIED_COMMIT_SHA> \
  -t YOUR_DOCKERHUB_USER/catvton-runpod-worker:1.0.0 .
```

برای production، revision وزن‌ها را با env به commit معتبر Hugging Face pin کنید: `CATVTON_MODEL_REVISION=<commit>`. اگر repository/model به authentication نیاز داشت، secret را فقط در تنظیمات RunPod قرار دهید؛ آن را داخل image، Dockerfile یا Git commit نکنید.

## Docker build و push

```bash
docker build -t YOUR_DOCKERHUB_USER/catvton-runpod-worker:1.0.0 .
docker login
docker push YOUR_DOCKERHUB_USER/catvton-runpod-worker:1.0.0
```

قبل از push، smoke test را روی ماشین GPU انجام دهید. `latest` برای deploy قابل تکرار مناسب نیست؛ tag نسخه‌ای یا digest image را در RunPod استفاده کنید.

## Deploy روی RunPod Serverless

1. در RunPod Console به **Serverless > New Endpoint** بروید و نوع **Queue-based** را انتخاب کنید.
2. deployment source را Docker Registry قرار دهید و image tag یا ترجیحاً digest خود را وارد کنید. برای private image، registry credential را در RunPod ثبت کنید.
3. GPU را انتخاب کنید. CatVTON رسمی برای `768x1024` حدود 8 GB VRAM گزارش کرده است؛ برای margin مناسب AutoMasker و fragmentation، GPU با 16 GB یا بیشتر مثل RTX 4090/L4/A4000/A5000 انتخاب عملی‌تری است. GPUهای مجاز را بر اساس availability و قیمت حساب خود انتخاب کنید.
4. Container disk باید برای image و cache وزن‌ها فضای کافی داشته باشد (حداقل 20–30 GB پیشنهاد می‌شود). در صورت استفاده از network volume یا cached model، آن را روی `/models/huggingface` mount کنید.
5. timeout هر job را با اندازه تصاویر و GPU benchmark کنید؛ شروع مناسب `300` ثانیه است. idle timeout را طوری تنظیم کنید که هزینه و cold start متعادل شوند.
6. برای توسعه `min workers = 0` و `max workers = 1` مناسب است. برای production latency-sensitive، `min workers = 1` cold start را حذف می‌کند؛ `max workers` را بر اساس نرخ درخواست، quota و بودجه افزایش دهید. concurrency هر worker را `1` نگه دارید.
7. environment variableهای لازم را تنظیم کنید و endpoint را deploy کنید. startup log باید `CatVTON worker is ready` را نشان دهد.

Environment variableهای اختیاری:

| نام | پیش‌فرض | توضیح |
|---|---:|---|
| `OUTPUT_JPEG_QUALITY` | `90` | کیفیت JPEG از 1 تا 100 |
| `CATVTON_WIDTH` / `CATVTON_HEIGHT` | `768` / `1024` | مضرب مثبت 8 |
| `CATVTON_GUIDANCE_SCALE` | `2.5` | CFG pipeline رسمی |
| `CATVTON_SOURCE_DIR` | `/opt/CatVTON` | محل source رسمی |
| `CATVTON_MODEL_ID` | `zhengchong/CatVTON` | repo وزن‌ها |
| `CATVTON_MODEL_REVISION` | unset | commit/tag وزن‌ها |
| `CATVTON_BASE_MODEL` | `booksforcharlie/stable-diffusion-inpainting` | مدل پایه |
| `CATVTON_SKIP_SAFETY_CHECK` | `false` | فقط پس از ارزیابی policy تغییر دهید |
| `EMPTY_CUDA_CACHE_AFTER_JOB` | `false` | معمولاً false سریع‌تر است؛ در فشار حافظه true |
| `LOG_LEVEL` | `INFO` | سطح log |

### Endpoint ID و API Key

پس از deploy، Endpoint ID در صفحه endpoint و URL آن دیده می‌شود. API Key را از **RunPod Settings > API Keys** بسازید. آن را فقط در secret manager یا environment پروژه مصرف‌کننده نگه دارید:

```bash
export RUNPOD_ENDPOINT_ID="..."
export RUNPOD_API_KEY="..."
```

### درخواست curl

برای جلوگیری از محدودیت command line، payload را در `request.json` قرار دهید. base64 باید بدون newline باشد:

```bash
PERSON_B64=$(base64 -w 0 samples/person.jpg)
GARMENT_B64=$(base64 -w 0 samples/garment.jpg)

jq -n \
  --arg person "$PERSON_B64" \
  --arg garment "$GARMENT_B64" \
  '{input: {person_image_base64: $person, garment_image_base64: $garment, garment_category: "upper_body", prompt: "", seed: 42, steps: 30}}' \
  > request.json

curl --request POST \
  "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header "Content-Type: application/json" \
  --data-binary @request.json
```

RunPod برای `/run` محدودیت payload حدود 10 MB و برای `/runsync` حدود 20 MB دارد. با اینکه worker برای **هر تصویر** سقف 10 MiB دارد، دو تصویر base64 ممکن است از سقف endpoint عبور کنند. برای تصاویر بزرگ resize/compress سمت client ضروری است؛ برای بارهای بزرگ‌تر باید معماری object storage + URL امن/کوتاه‌عمر را به‌عنوان توسعه جداگانه اضافه کرد.

برای job طولانی، `/run` را صدا بزنید و `id` برگشتی را از `/status/JOB_ID` poll کنید. Telegram Bot نیز بهتر است async endpoint را استفاده کند تا webhook یا update loop مسدود نشود.

## اتصال Telegram Bot

این worker مستقل می‌ماند. Bot باید در repository خودش:

1. دو فایل Telegram را download، در صورت نیاز resize و JPEG-compress، و base64 کند.
2. payload بالا را با `RUNPOD_ENDPOINT_ID` و `RUNPOD_API_KEY` که در secret manager نگهداری شده ارسال کند.
3. برای `/run` وضعیت job را با backoff و timeout poll کند.
4. `output.image_base64` را decode کرده و bytes JPEG را با متد ارسال تصویر Telegram بفرستد.
5. روی `status=failed`، code را برای telemetry ثبت کند ولی پیام فنی/secret را به کاربر نشان ندهد.

base64، تصاویر و Authorization header نباید در logهای Bot ثبت شوند. timeout شبکه Bot باید از timeout endpoint کمتر نباشد و retry تنها برای خطاهای transient با idempotency مناسب انجام شود.

## Cold start و عملیات production

- دانلود وزن‌ها در startup می‌تواند چندین دقیقه طول بکشد. image baked، RunPod model caching/network volume، و `min workers=1` این زمان را کاهش می‌دهند.
- scale-to-zero هزینه را کم می‌کند اما اولین درخواست latency زیادی خواهد داشت.
- startup را با log readiness مانیتور کنید؛ handler تا پایان model loading ثبت نمی‌شود.
- زمان inference، total time، OOM، startup duration و queue delay را مانیتور کنید. هیچ تصویر یا base64 را به telemetry نفرستید.
- `OUTPUT_JPEG_QUALITY` و resize سمت Bot را طوری تنظیم کنید که پاسخ زیر محدودیت payload RunPod و Telegram بماند.
- تغییر CUDA/PyTorch/CatVTON ref را ابتدا روی همان GPU class staging تست کنید.

## هشدار بسیار مهم مجوز

مجوز **کد این worker**، **کد CatVTON**، **وزن‌های CatVTON**، **مدل پایه Stable Diffusion/VAE**، **DensePose/SCHP** و **دیتاست‌هایی که برای آموزش استفاده شده‌اند** موضوعات جداگانه‌اند و باید جداگانه بررسی شوند. repository رسمی CatVTON در حال حاضر مواد خود را با محدودیت non-commercial معرفی می‌کند؛ این README هیچ ادعایی درباره مجاز بودن استفاده تجاری ندارد.

پیش از هر استفاده تجاری یا ارائه سرویس عمومی، متن دقیق و نسخه جاری مجوز تمام dependencyها، sourceها، model weights، base modelها و dataset terms را با مشاور حقوقی تأیید کنید. تغییر یا نبودن یک فایل License در این worker مجوز upstream را تغییر نمی‌دهد.

## منابع upstream

- [Official CatVTON repository](https://github.com/Zheng-Chong/CatVTON)
- [CatVTON model weights](https://huggingface.co/zhengchong/CatVTON)
- [RunPod handler documentation](https://docs.runpod.io/serverless/workers/handler-functions)
- [RunPod request documentation](https://docs.runpod.io/serverless/endpoints/send-requests)
