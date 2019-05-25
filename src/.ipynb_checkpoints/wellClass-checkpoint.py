import lasio

import pandas

class Well:
    
    def __init__(self, filename):

        #готовой функцией библиотеки lasio считываем лас-файл, далее будем из него вытягивать данные
        self.lasObject = lasio.read(filename)#, autodetect_encoding = True)
       
		
        #считываем с полученного лас-объекта все кривые(геофизические методы ГК НГК....)
        self.curves = {curve.mnemonic : curve.data for curve in self.lasObject.curves}

        #считываем с лас-объекта имя скважины
        self.name = str(self.lasObject.well["WELL"].value) 

        PZ_names = ["KS", "КС", "ПЗ", "pz"]
        for name in PZ_names:
            if name in self.curves.keys():
                self.curves["PZ"] = self.curves[name]
                
        PS_names = ["SP", "ПС", "ps"]
        for name in PS_names:
            if name in self.curves.keys():
                self.curves["PS"] = self.curves[name]
                
        DS_names = ["CALI", "ДС", "KV", "ds"]
        for name in DS_names:
            if name in self.curves.keys():
                self.curves["DS"] = self.curves[name]
                
        GK_names = ["GR", "ГК", "gk"]
        for name in GK_names:
            if name in self.curves.keys():
                self.curves["GK"] = self.curves[name]
                
        NGK_names = ["NGL", "НГК", "NKDT", "JB", "JM", "ngk"]
        for name in NGK_names:
            if name in self.curves.keys():
                self.curves["NGK"] = self.curves[name]

        if "MD" in self.curves.keys():
            self.curves["DEPT"] = self.curves["MD"]