from util_scripts.labelling import assigning_labels
from util_scripts.fonts_vector import fonts2vec
from util_scripts.merge_grobid_and_pdf_alto import merge_using_pdfalto
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
from joblib import Parallel, delayed
from tqdm import tqdm
import sqlite3
import os


class preprocessed_dataframe:
    def __init__(self, sets, excel_file, n_jobs=1, save_state=True, database=None):
        # give the information in the data format
        self.sets = sets
        self.save_state = save_state
        self.excel_file = excel_file
        self.n_jobs = n_jobs
        self.database = database

    def database_connect(self):
        """this function will make the database and add the csv to the the database"""
        # intialize schema or load the database file if db already exist
        if self.database is not None:
            conn = sqlite3.connect(self.database)
            cursor = conn.cursor()
            return conn, cursor

    def get_font_vectors(
        self, df, pdf_alto_xml_main, path, params, vectorized_fonts, save_state=True
    ):
        font_vectors = []
        for font_lines in df["fonts"]:
            # print(font_lines)
            font_para_vector = None
            for font_line in font_lines:
                fonts = font_line.split()
                count = 0
                fontline_vector = None
                for font in fonts:
                    try:
                        # print(font)
                        font_no = int(font[4:])
                        row = vectorized_fonts.iloc[font_no, 1:].values.copy()

                        count += 1
                    except:
                        continue
                    if fontline_vector is None:
                        fontline_vector = row
                    else:
                        fontline_vector += row
                try:
                    fontline_vector = fontline_vector / count
                except:
                    # if some how we cant get the font line vector because the font is not present or some other reason
                    # then assign every thing to 0's
                    fontline_vector = np.zeros(15)
                if font_para_vector is None:
                    font_para_vector = fontline_vector
                else:
                    font_para_vector += fontline_vector
            font_para_vector = font_para_vector / len(font_lines)
            font_vectors.append(font_para_vector)

        fonts = pd.DataFrame(font_vectors, columns=params)

        # print(params)
        # display(fonts)

        combined = pd.concat([df, fonts], axis=1)

        pdf_file = "/".join(
            pdf_alto_xml_main.replace(".xml", ".pdf").rsplit("/", 2)[1:]
        )

        combined["pdf_path"] = pdf_file

        if self.save_state is True:
            combined.to_csv(path + "/" + "data.csv")  # add self.path
            # print(path)

        return combined

    def local_to_global_font_map(self, pdf_alto_xml_main):
        pdfalto_xml = pdf_alto_xml_main
        with open(pdfalto_xml, "r") as file:
            # Read each line in the file, readlines() returns a list of lines
            content = file.readlines()
            # Combine the lines in the list into a string
            # print(content)
            content = "".join(content)
            bs_content = bs(content, "xml")
            file.close()
        style_info = bs_content.find_all("Styles")

        mapping_local_global = {}
        for element in style_info[0]:
            mapping_local_global[element.get("ID")] = element.get("FONTFAMILY")

        return mapping_local_global

    def process_one(
            self,
            grobid_xml,
            labels_xml,
            pdf_alto_xml_main,
            sample_pdf,
            excel_file,
            save_state,
    ):
        # пропускаем уже обработанные
        if os.path.exists(os.path.join(grobid_xml.rsplit("/", 1)[0], "data.csv")):
            return ("skipped", grobid_xml, None)

        try:
            # builds a table from grobid + assigns labels from extthm annotations
            table, scales = assigning_labels(
                show_images=False, grobid_xml=grobid_xml, labels_xml=labels_xml
            ).fit()

            # get the font_vectors table
            vectorized_fonts = fonts2vec(
                pdf_alto_xml_main, excel_file=excel_file
            ).get_dataframe()

            # get the local to global mapping
            mappings = self.local_to_global_font_map(pdf_alto_xml_main)

            params = vectorized_fonts.columns[1:]  # get all column names

            # merge GROBID boxes with ALTO lines (to attach fonts to each box)
            temp = merge_using_pdfalto(
                xml_main=pdf_alto_xml_main,
                sample_pdf=sample_pdf,
                scales=scales,
                table=table,
            ).merge()

            # ---- далее идёт код, который был ниже в оригинале ----
            def para_to_global(para):
                para_fonts = []
                for line in para:
                    line_gfont = ""
                    if len(line) != 0:
                        for font in line.split():
                            line_gfont += mappings[font] + " "
                    if len(line_gfont) != 0:
                        para_fonts.append(line_gfont)
                return para_fonts

            path = grobid_xml.rsplit("/", 1)[0]
            temp["global_fonts"] = temp["fonts"].apply(para_to_global)

            temp = self.get_font_vectors(
                temp,
                pdf_alto_xml_main,
                path,
                save_state=self.save_state,
                params=params,
                vectorized_fonts=vectorized_fonts,
            )
            # ----------------------------------------------------

            return ("ok", grobid_xml, temp)

        except Exception as e:
            # логируем проблемную статью, чтобы не потерять её из виду
            log_path = os.path.join(
                grobid_xml.rsplit("/", 1)[0].rsplit("/", 1)[0],
                "failed_articles.log",
            )
            with open(log_path, "a", encoding="utf8") as log:
                log.write(
                    f"{grobid_xml}\t{type(e).__name__}\t{e}\n"
                )
            print(f"[SKIP] {grobid_xml}: {type(e).__name__}: {e}")
            return ("failed", grobid_xml, str(e))

        def para_to_global(para):
            para_fonts = []
            for line in para:
                line_gfont = ""
                if len(line) != 0:
                    for font in line.split():
                        line_gfont += mappings[font] + " "
                if len(line_gfont) != 0:
                    para_fonts.append(line_gfont)

            return para_fonts

        path = grobid_xml.rsplit("/", 1)[0]
        temp["global_fonts"] = temp["fonts"].apply(para_to_global)
        # print("1",temp.shape)
        temp = self.get_font_vectors(
            temp,
            pdf_alto_xml_main,
            path,
            save_state=self.save_state,
            params=params,
            vectorized_fonts=vectorized_fonts,
        )
        # print("2",temp.shape)
        # temp= temp.applymap(str) ###

        # return temp
        # except Exception as error:
        #     print(error)
        #     print("an error occured !{}".format(grobid_xml))
        #     return

    def fit(self):
        res = Parallel(n_jobs=self.n_jobs, verbose=20)(
            delayed(self.process_one)(
                data[0],
                data[1],
                data[2],
                data[3],
                excel_file=self.excel_file,
                save_state=self.save_state,
            )
            for data in tqdm(self.sets)
        )

        return res
