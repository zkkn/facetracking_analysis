# [WIP] Face Tracking and Analysis
顔検知・トラッキング・属性分析（年齢と性別）をraspberry piとneural compute stick 2で実行するコードです。
現在、ロジックは完成しつつもraspberry piで一度も実行したことのない状態なのでバグ大ありコードになっているので、注意してください。

## HOW TO USE
raspberry pi v3 with neural compute stick 2
```
python face_tracker_analysis.py -i YOUR_VIDEO_PATH -d MYRIAD -d_ag MYRIAD
```

## 参考
### [AIを始めよう！PythonでOpenVINOの仕組みを理解する](https://qiita.com/ammo0613/items/ff7452f2c7fab36b2efc)
OpenVINO特有の推論の書き方を最小コードで理解できるので、既存のプロジェクトのコードを読む際の前提知識として大変良い。
もっと早く読むべきであった。

### [Object Detection MobileNet-SSD on NCS2 with Raspberry Pi](https://github.com/kodamap/object_detection_demo)
Detection
- object detection
- face detection
Analysis
- age and gender
- emotion
- head position

上記の推論の種類をflask経由でブラウザで結果を見れる。(素晴らしい！！）
クラス設計含めて大変参考になった。

