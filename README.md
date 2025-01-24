# Projeto: Predição de Inadimplência (*Default*) 

![capa](https://img.freepik.com/vetores-premium/conceito-de-especialista-financeiro-com-especialista-em-mulher_450309-518.jpg?w=740)


### **1. Introdução**  

A X-Health, uma empresa do setor B2B de dispositivos eletrônicos para saúde, enfrenta desafios relacionados à inadimplência de clientes em vendas a crédito. A falta de pagamento impacta diretamente o fluxo de caixa e a sustentabilidade financeira da empresa, exigindo uma solução preditiva para antecipar quais clientes possuem maior risco de não pagamento. Para mitigar esse risco, foi desenvolvido um modelo preditivo de default, capaz de antecipar clientes com alta probabilidade de não pagamento (*default*).

A solução proposta utiliza uma abordagem baseada em Machine Learning. O modelo foi treinado e validado com um conjunto de dados estruturado, considerando variáveis representativas do comportamento financeiro dos clientes, incluindo histórico de pagamentos, perfil tributário, características empresariais e informações de bureaus de crédito. Foram testados diferentes algoritmos, incluindo Regressão Logística, Random Forest e XGBoost, sendo este último escolhido por apresentar o melhor equilíbrio entre métricas de desempenho, como recall e precisão.

O modelo final é integrado a uma função de predição. Esta solução tem potencial para reduzir a exposição ao risco financeiro da empresa, garantindo uma gestão de crédito mais eficiente e baseada em dados.


### **2. Objetivo**   


*  O objetivo deste projeto é desenvolver um modelo preditivo de *default*, permitindo que o time financeiro tome decisões mais assertivas.

Com essa previsão, a empresa poderá: 

*  Reforçar critérios de crédito para clientes de alto risco.
*  Oferecer condições diferenciadas para clientes que possuem baixo risco de inadimplência.
*  Mitigar prejuízos financeiros, garantindo um fluxo de caixa mais saudável.


### **3. O dataset** 


*  Cada linha representa um cliente e as colunas representam diferentes tipos de informações desses clientes. A variável alvo é `default`, que indica se o cliente veio a se tornar inadimplente (1) ou não (0). As variáveis do dataset são descritas abaixo:

    -  `default_3months` : Quantidade de default nos últimos 3 meses
    -  `ioi_36months`: Intervalo médio entre pedidos (em dias) nos últimos 36 meses
    -  `ioi_3months`: Intervalo médio entre pedidos (em dias) nos últimos 3 meses
    -  `valor_por_vencer`: Total em pagamentos a vencer do cliente B2B, em Reais
    -  `valor_vencido`: Total em pagamentos vencidos do cliente B2B, em Reais
    -  `valor_quitado`: Total (em Reais) pago no histórico de compras do cliente B2B 
    -  `quant_protestos` : Quantidade de protestos de títulos de pagamento apresentados no Serasa
    -  `valor_protestos` : Valor total (em Reais) dos protestos de títulos de pagamento apresentados no Serasa
    - `quant_acao_judicial` : Quantidade de ações judiciais apresentadas pelo Serasa
    -  `acao_judicial_valor` : Valor total das ações judiciais (Serasa) 
    -  `participacao_falencia_valor` : Valor total (em Reais) de falências apresentadas pelo Serasa
    -  `dividas_vencidas_valor` : Valor total de dívidas vencidas (Serasa)
    -  `dividas_vencidas_qtd` : Quantidade total de dívidas vencidas (Serasa)
    -  `falencia_concordata_qtd` : Quantidade de concordatas (Serasa)
    -  `tipo_sociedade` : Tipo de sociedade do cliente B2B 
    -  `opcao_tributaria` : Opção tributária do cliente B2B
    -  `atividade_principal` : Atividade principal do cliente B2B
    -  `forma_pagamento` : Forma de pagamento combinada para o pedido
    -  `valor_total_pedido` : Valor total (em Reais) do pedido em questão
    -  `month` : Mês do pedido
    -  `year` : Ano do pedido
    -  `default` : Status do pedido: default = 0 (pago em dia), default = 1 (pagamento não-realizado)


### **4. Metodologia** 


*  1.  Análise Exploratória dos dados (Notebook 1)
    *  Identificação de padrões e insights

*  2.  Modelagem (Notebook 2)
    *  Limpeza dos dados
    *  Construção do Pipeline (FeatureEngineeringTransformer + ColumnTransformer + Classificador)
    *  Treinamento de algoritmos (Logistic Regression, RandomForest, XGBoost) e comparação de métricas
    *  Escolha de melhor modelo 
    *  Ajuste de Hiperparâmetros via RandomizedSearchCV, selecionando o melhor conjunto de parâmetros.
    *  Validação do modelo e checagem de overfitting 

*  3.  Função de Predição (Notebook 3) 
    *  Implementa uma função que: 
        *  Recebe um dicionário com as informações do cliente 
        *  Retorna {"default": 0 or 1} (previsão de *default*)


#### Transformação dos dados

*  Para centralizar e organizar todas as transformações aplicadas aos dados, foi criado um Custom Transformer (FeatureEngineeringTransformer), que foi salvo no diretório src/.

*  Esse transformer inclui as seguintes etapas:

    *  Criação de novas features (ex : log-transform para valores financeiros, proporção de valores quitados/vencidos, score de risco).
    *  Agrupamento de categorias (ex : tipo de sociedade e setor de atividade).
    *  Conversões de tipos (binarização de colunas, transformação de floats para inteiros).
    *  Manutenção de consistência - Garantia de que novos dados recebam as mesmas transformações aplicadas no treino.


#### Detalhes do Modelo

*  Modelo escolhido: O **XGBoost** foi o modelo final escolhido, pois apresentou melhor performance em recall, que é uma métrica relevante para minimizar falsos negativos (ou seja, clientes que não fariam o pagamento, mas não foram identificados pelo modelo). O custo de um falso negativo pode ser muito alto para a empresa. Assim, garantir que o modelo identifique o maior número possível de clientes inadimplentes foi a prioridade.

*  Performance: AUC= 0.89 - Recall ~80% - Precisão ~50%


#### **5. Como utilizar**  


*  1.  Clonar este repositório



    ```bash
    git clone https://github.com/MaiaraMoro/X-health/

    cd X-health

    ```
    
    
*  2.  Instalar dependências (arquivo environment.yml) 



    ```bash
    conda env create -f environment.yml
    ```


*  3.  Rodar Notebooks
    * Executar o Notebook 1 (notebooks/01_analise_exploratoria.ipynb): Análise Exploratória.
    *  Executar o Notebook 2 (notebooks/02_pipeline_modelagem.ipynb): treinamento e avaliação do modelo. Um pipeline final é salvo em model/pipeline_predicao_default.pkl.
    *  Executar o Notebook 3 (notebooks/03_funcao_predicao.ipynb): função de predição *predict_default()*


#### Contato


*  📧 : maimoro98@gmail.com
