# CNNのC++実装

## 必要なパッケージ
**C++**
* google-test : <http://opencv.jp/googletestdocs/primer.html>
* protobuf : <https://github.com/google/protobuf/tree/master/src>

**Python** (重みやデータの生成用）
* numpy
* pickle
* scikit-learn

## ビルド方法
**ALL**
```
$ make -j4
```
**MLP**
```
$ make bin/mlp

$ ./bin/mlp test  # 推論の場合

$ ./bin/mlp train  # 学習の場合
``` 
**CNN**
```
$ make bin/cnn

$ ./bin/cnn test  # 推論の場合

$ ./bin/cnn train  # 学習の場合
```
**テスト**
```
$ make bin/utest

$ ./bin/utest
```
**生成されたファイルの削除**
```
$ make clean
```

※Stack Overflowした場合
```
$ ulimit -s 16384
```

## コード構成
### src/util
* function.hpp : 活性化関数や畳込み、プーリングなど基本的な関数が実装されている
* tensor.hpp : 多次元配列を扱うTensorクラスを定義している
* read\_data.cpp, read\_data.hpp : MNISTのデータの読み込みをするクラス

### src/cnn
* cnn.hpp : CNNクラスを定義している

### src/mlp
* mlp.hpp : MLPクラスを定義している

### src/python
* make\_cnn\_weight.py : cnnの重みを定義するcnn_weight.hppを生成するファイル
* make\_mlp\_weight.py : mlpの重みを定義するmlp_weight.hppを生成するファイル
* make\_mnist\_data.py : mnistのデータを定義するmnist_data.hppを生成するファイル
* make\_test\_array.py : テスト用の配列を定義するtest_array.hppを生成するファイル

### test/
* util_test.cpp : いろいろな関数のテストが実装されている

## 実装上のルール
### For FPGA
* std::vector, std::arrayはVivado HLSでサポートされていないので原則Tensorクラスか配列を使う
* malloc, freeを使わず、メモリ確保はtemplateを使うなどして静的に行う
* Vivado HLSでサポートされていないものを使うときはコメントを入れる
* templateを多様することになるため基本的に実装はすべてヘッダファイルに書くことになるはず

### For Maintainability
* 関数を実装したらテストを書く
* マージする前にテストが通っていることを確認する（CIとかと連携させてテストが通らないとマージできない設定にしたい...）
* pull requestはできるだけ細かくする
* branchも細かく切る
* コードレビューしてもらってからマージ
* スタイルはそんなに厳しく見ないけど、基本的には<https://ttsuki.github.io/styleguide/cppguide.ja.html>に従う
* 基本的にメンバ変数はprivateにする

### For Performance
* 関数内で値が変更される場合はポインタで渡す
* 関数内で値が変更されない場合は参照渡しにする
* **型にできるだけ依存しないような実装**をする

## Debug tips
通常のプリントデバッグやデバッガでのデバッグに加えてValgrindを使うと効果的だったりする。

**Valgrind**

メモリリークとかをチェックしてくれるツール

公式: <http://valgrind.org/>

日本語Wiki: <https://ja.wikipedia.org/wiki/Valgrind>

コマンド：　基本これだけ使っておけばなんとかなる
```
valgrind --leak-check=full ./a.out
```

