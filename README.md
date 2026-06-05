## What Information Does YOLO Read from the YAML File?

### 1. Dataset Paths
YOLO needs to know where the training and validation images are stored.

```yaml
train: images/train
val: images/val
```

These paths allow YOLO to locate and load the dataset during training. Without these paths, YOLO would not know where the images are stored.

---

### 2. Class Names
YOLO annotations use numerical IDs rather than text labels. Example label file:

```
0 0.45 0.52 0.20 0.30
1 0.60 0.40 0.15 0.25
```

The YAML file defines what these IDs represent:

```yaml
names:
  0: person
  1: car
```

Which means `0 → person` and `1 → car`. Without the YAML file, YOLO would only know the class IDs and would not be able to associate them with meaningful class names.

---

### 3. Number of Classes
YOLO's output layer is built according to the number of object classes.

```yaml
nc: 3
```

This tells YOLO that the dataset contains three classes: `person`, `car`, and `bicycle`. If the number of classes is incorrect, training results will be incorrect or training may fail.

---

## Why JPG Images Alone Are Not Enough

A JPG image only contains pixel values. The image does not contain information about:

- What objects are present
- How many objects exist
- Where the objects are located
- Which class each object belongs to

Humans can easily recognize objects in an image, but YOLO cannot automatically determine the object category or bounding box coordinates — this information must be provided through annotations.

---

## YOLO Label Files

For every image, YOLO expects a corresponding `.txt` annotation file:

```
dog.jpg
dog.txt
```

The label file contains object information in the following format:

```
0 0.50 0.50 0.40 0.30
```

**YOLO annotation format:**

```
<class_id> <x_center> <y_center> <width> <height>
```

| Field | Description |
|-------|-------------|
| `class_id` | Object category |
| `x_center` | Center x-coordinate of the object |
| `y_center` | Center y-coordinate of the object |
| `width` | Bounding box width |
| `height` | Bounding box height |

> All coordinate values are normalized between `0` and `1`.
