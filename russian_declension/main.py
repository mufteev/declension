"""Точка входа: python -m russian_declension.main"""
import logging, uvicorn
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def main():
    uvicorn.run("russian_declension.api.app:app",
                host="0.0.0.0", port=8000, reload=False, log_level="info")

if __name__ == "__main__":
    main()
