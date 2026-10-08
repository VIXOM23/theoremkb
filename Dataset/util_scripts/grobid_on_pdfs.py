import os
import subprocess
import time
from tqdm import tqdm
from joblib import Parallel, delayed
from util_scripts.grobid_client_python.grobid_client.grobid_client import GrobidClient


class grobid_on_pdfs:
    def __init__(
        self,
        grobid_client,
        sub_folders,
        n_jobs,
        docker_image="grobid/grobid:0.8.0",
        container_name="grobid",
        grobid_server="http://localhost:8070",
        config_path=None,
        start_container=True,
    ):
        """
        grobid_client  : путь к папке grobid-client-python (там, где лежит config.json)
        sub_folders    : список папок с PDF-файлами
        n_jobs         : число параллельных задач
        docker_image   : образ Grobid в Docker Hub
        container_name : имя контейнера (чтобы потом можно было убить/переиспользовать)
        grobid_server  : адрес REST API Grobid внутри Docker
        config_path    : путь к config.json (если None, берётся из grobid_client)
        start_container: если True — поднять контейнер самому, иначе считать, что он уже запущен
        """
        self.grobid_client = grobid_client
        self.sub_folders = sub_folders
        self.n_jobs = n_jobs
        self.docker_image = docker_image
        self.container_name = container_name
        self.grobid_server = grobid_server
        self.config_path = config_path or os.path.join(grobid_client, "config.json")
        self.start_container = start_container

    # ---------- Docker helpers ----------
    def _is_container_running(self):
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={self.container_name}",
             "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        return self.container_name in result.stdout.split()

    def _start_docker(self):
        # Если контейнер уже есть (но остановлен) — удаляем, чтобы не конфликтовать
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True, text=True,
        )
        print(f"🐳 Запускаю Grobid в Docker ({self.docker_image})...")
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", self.container_name,
                "-p", "8070:8070",
                "--rm",
                self.docker_image,
            ],
            check=True,
        )

    def _wait_for_grobid(self, timeout=180, interval=5):
        """Ждём, пока Grobid поднимется и начнёт отвечать."""
        import urllib.request
        import urllib.error
        url = self.grobid_server + "/api/isalive"
        start = time.time()
        while time.time() - start < timeout:
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    if r.status == 200:
                        print("✅ Grobid готов.")
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(interval)
        raise RuntimeError(f"Grobid не поднялся за {timeout} секунд.")

    def _stop_docker(self):
        print("🛑 Останавливаю контейнер Grobid...")
        subprocess.run(
            ["docker", "stop", self.container_name],
            capture_output=True, text=True,
        )

    # ---------- Main ----------
    def fit(self):
        valid_paths = self.sub_folders.copy()

        # Пропускаем папки, где уже есть .tei.xml
        for pdf_path in tqdm(self.sub_folders, desc="Проверка папок"):
            try:
                tei_xml = [x for x in os.listdir(pdf_path) if x.endswith(".tei.xml")]
            except (FileNotFoundError, NotADirectoryError):
                continue
            if len(tei_xml) == 1:
                valid_paths.remove(pdf_path)

        if not valid_paths:
            print("Все папки уже обработаны — нечего делать.")
            return

        if self.start_container:
            if not self._is_container_running():
                self._start_docker()
            else:
                print(f"ℹ️  Контейнер '{self.container_name}' уже запущен.")
            self._wait_for_grobid()

        # 2. Подключиться клиентом к Docker Grobid
        client = GrobidClient(config_path=self.config_path)

        try:
            Parallel(n_jobs=self.n_jobs)(
                delayed(client.process)(
                    "processFulltextDocument",
                    folder,
                    n=20,
                    generateIDs=True,
                    consolidate_citations=True,
                    tei_coordinates=True,
                    segment_sentences=True,
                    include_raw_affiliations=True,
                    include_raw_citations=True,
                )
                for folder in tqdm(valid_paths, desc="Grobid processing")
            )
        finally:
            # 3. Погасить контейнер
            if self.start_container:
                self._stop_docker()


# ---------- Пример использования ----------
if __name__ == "__main__":
    grobid_client = "/Users/mike/grobid-client-python-0.2.0"
    destination = "/Users/mike/articles"

    sub_folders = [
        os.path.join(destination, element)
        for element in os.listdir(destination)
    ]

    grobid_on_pdfs(
        grobid_client=grobid_client,
        sub_folders=sub_folders,
        n_jobs=4,
    ).fit()