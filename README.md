# takenaka_wellness

05*project*竹中様\_ウェルネス空調制御

こちらのプログラムは `Python 3.10` で動作確認済みです。

## 環境構築：

リポジトリをクローン

```
git clone https://github.com/MENTERU/takenaka_wellness.git
cd takenaka_wellness
```

`uv` をインストール(uv がない場合):

```
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

`uv` で仮想環境を作成(Python 3.11 を使用):

```
uv venv --python 3.10
```

`uv` で依存パッケージをインストール:

```
uv sync
```

## main.py を実行：

### ローカルで

```
uv run main.py
```

## DB のアクセス：

`db_utils/private_info.py`を作成し、下記の内容を追加してください。

```
DB_CONFIGS = {
    "host1": {
        "dbname": "<DB_NAME>",
        "user": "<USERNAME>",
        "password": "<PASSWORD>",
        "host": "<HOST>",
        "port": "<PORT>",
    },
    "host2": {
        "dbname": "<DB_NAME>",
        "user": "<USERNAME>",
        "password": "<PASSWORD>",
        "host": "<HOST>",
        "port": "<PORT>",
    },
}
```

ここに適切な値を入力してください:

- &lt;DB_NAME&gt;: データベース名
- &lt;USERNAME&gt;: データベースのユーザー名
- &lt;PASSWORD&gt;: データベースのパスワード
- &lt;HOST&gt;: データベースのホスト
- &lt;PORT&gt;: データベースのポート番号

## フォルダ構成：

\*追加する必要ありフォルダをご注意、また、フォルダ構成もご注意ください。

```
├──Takenaka-Wellness          #追加する必要あり
│   ├── マスタデータ.xlsx 　　   #追加する必要あり
│   ├── 期間設定.xlsx 　　　     #追加する必要あり
│   ├── 00_Data　　　　　　      #追加する必要あり
│   │   ├── 00_RawData　　     #追加する必要あり
│   │   ├── 01_PreProcessData #追加する必要あり
│   │   └── 03_Output         #追加する必要あり
    └── takenaka_wellness    # ここから下がプロジェクトリポジトリの構造
      ├── data_module         # データ処理や取得モジュール
      ├── optimization_module # 最適化モジュール
      └── prediction_module   # 予測モジュール
      ├── README.md           # 説明
      ├── main.py             # メインプログラム
      ├── requirements.txt    # 必要なパッケージ
      └── .gitignore          # 無視するファイル
      ├── pyproject.toml  # プロジェクト設定ファイル。Pythonの標準規格（PEP 518）に準拠し、`uv` や `pip-tools` などのツールがパッケージ管理やビルドに使用する。
      ├── uv.lock  # 依存関係のバージョンを記録し、環境の一貫性を保つためのuvロックファイル。
      ├── .python-version　# Pythonバージョン (`pyenv`や`uv`などのツールが自動的に適切なバージョンを選択できるようになる。)

```
