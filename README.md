# GenLocalNews — demonstrador

Demonstrador de jornalismo assistido por IA desenvolvido na dissertação de mestrado *Inteligência Artificial no Jornalismo Local: Das Atas Municipais à Proposta de Notícias* (Universidade da Beira Interior, 2026). Trabalha sobre segmentos previamente extraídos de atas municipais e contém apenas o fluxo final usado na dissertação:

1. ingestão de segmentos em JSON, JSONL ou CSV;
2. classificação binária de noticiabilidade com o modelo final selecionado;
3. predição dos sete valores-notícia e dos IIP direto e derivado;
4. ordenação dos segmentos por IIP direto dentro de cada ata;
5. seleção humana dos segmentos enviados para geração;
6. geração estruturada remota com o Llama 3.1 8B + LoRA selecionado;
7. revisão, edição, aprovação ou rejeição humanas;
8. exportação local das notícias aprovadas.

A decisão binária de noticiabilidade e o ranking são mantidos como saídas distintas: a primeira vem da cabeça binária do classificador; a segunda usa o IIP direto. A geração só é executada para segmentos explicitamente selecionados pela pessoa utilizadora.

A aplicação não contém treino, avaliação experimental, RAG, segmentação de atas nem publicação autónoma. Os clientes de APIs externas existem apenas para o gerador alternativo descrito mais abaixo.

Demonstração em vídeo: https://www.youtube.com/watch?v=6NNVH4ArKLE

## Instalação local

No Windows, com o ambiente virtual ativo:

```powershell
pip install -r requirements.txt
copy .env.example .env
python app.py
```

Interface:

```text
http://localhost:5000/
```

A interface operacional apresenta, por segmento, a decisão binária, probabilidade e limiar, os sete valores-notícia, IIP direto, IIP derivado, posição no ranking e texto de origem.

## Modelos

Os pesos dos modelos não estão neste repositório.

| Componente | Modelo | Onde |
|---|---|---|
| Noticiabilidade | XLM-RoBERTa Base multitarefa (full fine-tuning, semente 42, limiar 0,845) | https://huggingface.co/liaad/genlocalnews-newsworthiness-xlmr |
| Geração | `meta-llama/Llama-3.1-8B-Instruct` + adaptador LoRA (r = 16) | https://huggingface.co/liaad/genlocalnews-llama3.1-8b-lora |

### Classificação — local

O checkpoint deve ser instalado em:

```text
models/selected/classification/bundle/checkpoint/
```

```powershell
python scripts/install_selected_models.py --classification-bundle CAMINHO/DO/BUNDLE
```

### Geração — serviço remoto

O Llama 3.1 8B + LoRA corre num serviço de inferência separado (precisa de GPU). A aplicação chama o endereço definido em `generation.remote_url` no ficheiro `config/model_selection.json` (por omissão `http://127.0.0.1:8000`), através de `POST /generate` e `GET /health`.

Para levantar o serviço sem servidor próprio, abrir `colab/GenLocalNews_geracao_colab.ipynb` no Google Colab com GPU, executar as células por ordem e copiar o endereço `https://….trycloudflare.com` que aparece no fim para `remote_url`. É necessário ter aceitado as condições do Llama 3.1 no Hugging Face. O endereço do túnel muda em cada sessão.

## Endpoints principais

```text
POST /api/upload
GET  /api/integration/status
GET  /api/integration/generation/health
POST /api/integration/classify
POST /api/integration/generate
GET  /api/results/<run_id>
POST /api/review/bootstrap
GET  /api/review/<batch_id>
```

`/api/integration/status` valida a configuração. Para geração remota, `ready=true` significa que a configuração está completa; `/api/integration/generation/health` verifica se o serviço remoto está realmente alcançável.

## Estrutura

```text
app.py
api/config.py
backend/
  dataset_loader.py
  preprocessor.py
  integration/
    classification_service.py
    generation_service.py
    generation_prompts.py
    registry.py
    vendor_classification/
  human_review.py
  routes.py
frontend/
  integrated_pipeline.html
  revisao.html
colab/GenLocalNews_geracao_colab.ipynb
config/model_selection.json
models/selected/
outputs/
scripts/install_selected_models.py
```

## Testes

```powershell
pytest -q
```


## Gerador online temporário para a revisão de propostas de notícia

Quando o serviço Llama 3.1 8B + LoRA não estiver acessível, a interface pode usar apenas os fornecedores online para os quais existiam credenciais na aplicação de apoio fornecida: IA Edu, Gemini 2.5 Pro, Llama 3.3 70B via Together e Llama 3.3 70B via Groq.

Esta alternativa existe exclusivamente para operacionalizar e testar o fluxo até à revisão de propostas de notícia. As propostas de notícia geradas por estes fornecedores ficam registadas com `generation_role=operational_fallback` e `experimental_status=not_selected_thesis_model`; não substituem o modelo generativo selecionado nem os resultados experimentais da dissertação.

As credenciais ficam apenas em `.env` (ver `.env.example`), que permanece ignorado pelo Git. Nunca versionar ou publicar esse ficheiro.


## Acesso à demonstração

Todas as páginas e endpoints da API exigem uma sessão autenticada. A página `/login` disponibiliza duas contas simples de demonstração: `inesctec / inesctec` e `ubi / ubi`. O mecanismo é deliberadamente simples e deve ser substituído por autenticação institucional num cenário de produção.

## Unidade de entrada

A versão atual continua a receber ficheiros JSON, JSONL ou CSV com segmentos previamente delimitados. O carregamento de uma ata integral e a segmentação automática são uma extensão prevista, mas não estão implementados nesta versão; esta distinção é apresentada explicitamente na interface para não confundir uma ata completa com o conjunto de segmentos que alimenta o fluxo atual.

## Dados

O repositório não inclui dados do CitiLink-News. Uma amostra e a documentação do conjunto de dados estão em https://github.com/LIAAD/citilink-news.
