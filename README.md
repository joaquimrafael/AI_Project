# Aprendizado de máquina no aprimoramento da negociação de pares

Desenvolvido por
* Joaquim Rafael Mariano Prieto Pereira - 10408805
* Henrique Arabe Neres de Farias - 10410152
* Gustavo Matta - 10410154
* Lucas Trebacchetti Eiras - 10401973

## Sobre o Projeto

Este repositório contém o script e os dados para a Análise Exploratória de Dados (EDA) de ações da B3, focada na preparação para uma estratégia de negociação de pares usando Machine Learning.

## Arquivos Principais

*   `eda_script.py`: O script Python com a análise exploratória.
*   `petr3_4_min.csv`: Dados de exemplo da PETR3.
*   `bbdc3_4_min.csv`: Dados de exemplo da BBDC3.
*   `itau3_4_min.csv`: Dados de exemplo da ITAU3.

## Como Executar

Para rodar o script de EDA, certifique-se de ter as bibliotecas `pandas`, `matplotlib` e `seaborn` instaladas (`pip install pandas matplotlib seaborn`).

Em seguida, execute o script Python no terminal:

```bash
python eda_script.py

Para rodar o código final de ML (contido na pasta ml_pairs_trading) execute o comando:
```bash
python run_trading_strategy.py --sl --sl_model (modelo de ML) --data_path dataset/(nome_do_dataset).csv  --n_epochs 3

Para executar os plots de comparação após rodar os modelos:
```bash
python comparison.py --dataset (nome_do_dataset) --base_dir ./
