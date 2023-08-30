## Basic image object detection with CoreML or CPU

## Run with CoreML
```bash
$ USE_COREML=true go run main.go

Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Min Time: 17.401875ms, Max Time: 21.7065ms, Avg Time: 19.258691ms, Count: 5
50th: 18.485666ms, 90th: 21.7065ms, 99th: 21.7065ms
```

## Run with CPU
```bash
$ go run main.go

Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Min Time: 41.5205ms, Max Time: 58.348084ms, Avg Time: 46.154341ms, Count: 5
50th: 43.471958ms, 90th: 58.348084ms, 99th: 58.348084ms
```
