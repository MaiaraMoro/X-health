import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    '''
    Aplica transformações no dataframe
    - construção de novas features
    - binarização de variáveis
    - agrupamento de categorias
    - log transformação
    - remoção de colunas 

    '''

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()

        # arredondando dias e transformando em inteiros
        X["ioi_36months"] = X["ioi_36months"].round().astype(int)
        X["ioi_3months"] = X["ioi_3months"].round().astype(int)

        # Transformando float em inteiro (e não precisam arredondar)
        cols_para_int = ["default_3months", "quant_protestos", "quant_acao_judicial", "dividas_vencidas_qtd", "falencia_concordata_qtd", "month", "year"] 
        X[cols_para_int] = X[cols_para_int].astype(int)

        # binarizar as colunas com muitos zeros
        cols_binarizar = ["quant_protestos", "quant_acao_judicial", "dividas_vencidas_qtd", "falencia_concordata_qtd", "default_3months"]
        for col in cols_binarizar:
            X[f"{col}_bin"] = (X[col] > 0).astype(int)
        
        # criar colunas log transformadas para as variáveis financeiras
        cols_to_log = ["valor_por_vencer","valor_vencido","valor_quitado", "valor_protestos", "acao_judicial_valor", "dividas_vencidas_valor", "valor_total_pedido"]
        for col in cols_to_log:
            log_col = f"log1p_{col}"
            X[log_col] = np.log1p(X[col])

        # categorizar ioi_3months e ioi_36months 
        X['ioi_36m_cat'] = pd.cut(X['ioi_36months'], bins=[0,20,38,70,9999], labels=["<=20", "21-38", "39-70", ">70"])
        X['ioi_3_cat'] = pd.cut(X['ioi_3months'], bins=[0,10,15, 20, 999], labels=["<=10", "11-15", "16-20", ">20"])

        # criar coluna com a razão entre ioi_3months e ioi_36months 
        X["ratio_ioi"] = X["ioi_3months"] / (X["ioi_36months"] + 1)

        # agrupar categorias em tipo_sociedade por características semelhantes
        grupos_sociedade = {'empresario (individual)': 'individual',
                            'empresa individual respons limitada empresaria': 'individual',
                            'empresario-mei(microempreendedor individual)': 'individual',
                            'sociedade empresaria limitada': 'sociedade empresarial',
                            'sociedade anonima fechada': 'sociedade empresarial',
                            'sociedade anonima aberta': 'sociedade empresarial',
                            'sociedade de economia mista': 'sociedade empresarial',
                            'sociedade simples pura': 'sociedade simples',
                            'sociedade simples limitada': 'sociedade simples',
                            'cooperativa': 'cooperativa',
                            'entidade sindical': 'outros',
                            'municipio': 'outros',
                            'servico social autonomo': 'outros',
                            'organizacao religiosa': 'outros',
                            'fundacao privada': 'outros',
                            'outras formas de associacao': 'outros'}
        X['tipo_sociedade_agrupado'] = X['tipo_sociedade'].map(grupos_sociedade)

        # extrair informacoes da forma de pagamento e criar novas features - n de parcelas, prazo medio e prazo maximo
        def extrair_metricas (valor):
            if pd.isna(valor) or valor.lower() == 'missing':
                return (np.nan, np.nan, np.nan)
            if valor.lower() == "boleto a vista":
                return (1, 0, 0)
            
            texto_limpo = re.sub(r"(boleto|dd|x|\(|\))", "", valor.lower())
            numeros = re.findall(r"\d+", texto_limpo)
            if not numeros:
                return (0,0,0)
            
            arr = list(map(int, numeros))
            return (len(arr), np.mean(arr), max(arr))
        X[["n_parcelas","prazo_medio","prazo_maximo"]] = X["forma_pagamento"].apply(extrair_metricas).apply(pd.Series)
        
        # classificar prazo maximo de pagamento em categorias (curto, medio, longo, muito longo)
        def classificar_forma_pto (prazo):
            if pd.isna(prazo):
                return "missing"
            elif prazo <= 30:
                return "curto"
            elif prazo <= 90:
                return "medio"
            elif prazo <= 180:
                return "longo"
            else:
                return "muito_longo"
        X['classificacao_prazo'] = X['prazo_maximo'].apply(classificar_forma_pto)

        # criar colunas binárias para indicar valores ausentes em opcao_tributaria e forma_pagamento 
        X["opcao_tributaria_missing"] = np.where(X["opcao_tributaria"] == "missing", 1, 0)
        X["forma_pagamento_missing"] = np.where(X["forma_pagamento"] == "missing", 1, 0)

        # agrupar atividades em categorias semelhantes e outras
        def categorizar_atividade(atividade):
            keywords = {
                "fundacao_ongs_instituicao": ["fundacao", "associacao", "instituicao", "fund", "sindicato"],
                "cooperativas": ["cooperativa"],
                "educacao": ["escola", "ensino", "curso", "cientifica", "especializacao"],
                "servicos": ["servico", "reparacao", "hospedagem", "consultoria", "repar", "borracharia", "locacao", "servicos", "serraria", "laboratorio", "serv"],
                "comercio": ["comercio", "venda", "mercado", "atacado", "com de", "supermercado", "papelaria", "bazar", "varejista", "farmacia", "concessionaria", "loja"],
                "industria": ["industria", "fabrica", "producao", "manufatura", "ind", "usinagem"]
            }
            for categoria, palavras in keywords.items():
                if any(palavra in atividade.lower() for palavra in palavras):
                    return categoria
            return "outros"
        X["atividade_agrupada"] = X["atividade_principal"].apply(categorizar_atividade)

        # proporção de valores vencidos e quitados
        X["prop_vencido"] = X.apply(lambda row: row["valor_vencido"] / (row["valor_quitado"] + row["valor_por_vencer"] + 1), axis=1)
        X["prop_quitado"] = X.apply(lambda row: row["valor_quitado"] / (row["valor_quitado"] + row["valor_por_vencer"] + 1), axis=1)

        # agrupar meses em trimestres
        def agrupar_meses(x):
            if x <= 4:
                return "1"
            elif x <= 6:
                return "2"
            elif x <= 9:
                return "3"
            else:
                return "4"
        X["trimestre"] = X["month"].apply(agrupar_meses)
            
        # remover colunas
        X = X.drop(["valor_por_vencer", "valor_vencido", "valor_quitado", "valor_protestos", "acao_judicial_valor", "dividas_vencidas_valor", "valor_total_pedido", 
                "quant_protestos", "quant_acao_judicial", "dividas_vencidas_qtd", "falencia_concordata_qtd", "forma_pagamento", "tipo_sociedade",
                "atividade_principal", "month"], axis=1)
        
        return X