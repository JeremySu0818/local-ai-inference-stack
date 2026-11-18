import base64
import io
import time


def encode_response(images, options, cancelled):
    data = []
    for image in images:
        if cancelled.is_set():
            return None
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", compress_level=1)
        data.append({"b64_json": base64.b64encode(buffer.getbuffer()).decode("ascii")})
    return {"created": int(time.time()), "seed": options.seed,
            "size": f"{options.width}x{options.height}", "data": data}
