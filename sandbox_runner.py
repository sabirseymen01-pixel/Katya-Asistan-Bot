"""
Katya Safe Execution Sandbox:
Katya'nın kendi kendine ürettiği Python kod parçacıklarını güvenli, izole bir kapsamda (scope)
çalıştıran, hata alması durumunda kendi hatasını yakalayıp düzeltmesine imkan tanıyan motor.
"""

import asyncio
import traceback

BANNED_IMPORTS = ["os.system", "subprocess", "shutil.rmtree", "__import__('os').system"]

async def execute_dynamic_code(code_str: str, execution_context: dict) -> dict:
    """
    Üretilen Python kodunu izole bir namespace içinde asenkron olarak icra eder.
    """
    for bad in BANNED_IMPORTS:
        if bad in code_str:
            return {"success": False, "error": f"Güvenlik kuralı ihlali: '{bad}' kullanılamaz."}

    local_vars = {**execution_context}
    
    # Kodun asenkron çalışabilmesi için fonksiyon kalıbı içine sarıyoruz
    indented_code = "\n".join([f"    {line}" for line in code_str.strip().split("\n")])
    wrapped_code = f"async def __katya_dynamic_action__():\n{indented_code}\n"

    try:
        exec(wrapped_code, local_vars)
        func = local_vars.get("__katya_dynamic_action__")
        
        # 15 saniyelik zaman aşımı (Timeout) kalkanı
        result = await asyncio.wait_for(func(), timeout=15.0)
        return {"success": True, "output": result}
    except asyncio.TimeoutError:
        return {"success": False, "error": "Kod çalışma zaman aşımına (15s) uğradı."}
    except Exception as e:
        tb = traceback.format_exc()
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}", "traceback": tb}
