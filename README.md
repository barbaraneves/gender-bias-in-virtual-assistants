# *Identifying Gender Bias in Virtual Assistants*

![alt text](resources/images/gender_bias.png)

Projeto Final da disciplina de Ciência de Dados - UFC 2021.1, em sua maioria feito no [Google Colab](https://research.google.com/colaboratory/faq.html). Colocamos em um repositório para facilitar o acesso e a correção.

---

## Objetivos
1. Desenvolver um projeto inédito de Ciência de Dados o mais próximo possível de um projeto real, a fim de demonstrar os conhecimentos adquiridos ao longo da disciplina.
2. Comparar os algoritmos escolhidos com conjuntos de dados reais utilizando métricas de avaliação vistas ou não na disciplina.

## Abstract

*A definir*

## Resultados 

| **Task** | **Dataset** | **Sample** | **Stratified Split** | **F1 LSTM** | **F1 BERT** | 
|  -------------- |  -----------  |  -----------  |  -----------  |----------- | ------------- | 
|  Tocixity, Multi-label  | Wikipedia Toxic Comments | Undersampling (~20k) | Yes | 0.67 | **0.68** | 
|  Gender Bias, Multiclass | MDGender | ~2k | Yes | 0.75 | **0.88** |
|  Gender Bias, Multiclass | ConvAI2  | 50k | Yes | 0.69 | **0.81**  |
|  Gender Bias, Multiclass  | LIGHT | 50k | Yes | 0.73 | **0.83**  |

---

## Quick Start

### Environment

Use o [`virtualenv`](https://virtualenv.pypa.io/en/latest/) para criar um ambiente Python.

```bash
virtualenv venv --python=python3

source venv/bin/activate
```

### Usage

Use o *package manager* [`pip`](https://pip.pypa.io/en/stable/) para instalar os pacotes necessários através do comando abaixo.

```bash
pip install -r requirements.txt
```

Depois, basta executar: 

```bash
jupyter notebook
```

## Visão Geral e Checkpoints

![ml canvas](/resources/images/ml_canvas.png)

Fornecemos abaixo os passos a serem seguidos para entendimento do projeto em forma de *checkpoints*.

* [Checkpoint 1 - Project Canvas](/resources/docs/CHECKPOINT_1_CANVAS.md)
* [Checkpoint 2 - Exploratory Data Analysis (EDA)](https://github.com/barbaraneves/gender-bias-in-virtual-assistants/tree/main/exploratory-data-analysis)
* [Checkpoint 3 - Data Preprocessing](https://github.com/barbaraneves/gender-bias-in-virtual-assistants/tree/main/data-preprocessing)
* [Checkpoint 4 - Models Training](https://github.com/barbaraneves/gender-bias-in-virtual-assistants/tree/main/models-training)
* [Checkpoint 5 - Models Evaluation](https://github.com/barbaraneves/gender-bias-in-virtual-assistants/tree/main/models-evaluation)
* [Final Checkpoint - Slide Presentation](/resources/docs/)

Na verdade, ao longo do desenvolvimento do trabalho, tivemos de entregar 3 *checkpoints*. Dos listados acima, os *checkpoints* oficiais são os 1 e 2, e os restantes dizem respeito mais a nossa organização interna. Nos avise se algo não estiver claro.

## Contato

Você pode enviar suas perguntas ou comentários para [Bárbara](https://github.com/barbaraneves), [Lucas](https://github.com/Lucas08Ben), [Samir](https://github.com/samirbraga) e [Vinicius](https://github.com/bgvinicius) :)
