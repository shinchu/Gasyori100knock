# Question 41-50

- 41. Cannyエッジ検出 (Step.1) エッジ強度
- 42. Cannyエッジ検出 (Step.2) 細線化
- 43. Cannyエッジ検出 (Step.3) ヒステリシス閾処理
- 44. Hough変換・直線検出 (Step.1) Hough変換
- 45. Hough変換・直線検出 (Step.2) NMS
- 46. Hough変換・直線検出 (Step.3) Hough逆変換
- 47. モルフォロジー処理(膨張)
- 48. モルフォロジー処理(収縮)
- 49. オープニング処理
- 50. クロージング処理


```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io

img_orig = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_256x256.png')
img_gray = cv2.cvtColor(img_orig, cv2.COLOR_RGB2GRAY)
img_128 = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_128x128.png')
img_gray_128 = cv2.cvtColor(img_128, cv2.COLOR_RGB2GRAY)
img_noise = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_256x256_noise.png')
img_dark = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_256x256_dark.png')
img_dark_gray = cv2.cvtColor(img_dark, cv2.COLOR_RGB2GRAY)
img_light = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/imori_256x256_light.png')
img_light_gray = cv2.cvtColor(img_light, cv2.COLOR_RGB2GRAY)

img_thorino = io.imread('https://yoyoyo-yo.github.io/Gasyori100knock/dataset/images/thorino.jpg')
img_thorino_gray = cv2.cvtColor(img_thorino, cv2.COLOR_RGB2GRAY)
```

# Q.41. Cannyエッジ検出 (Step.1) エッジ強度

Q.41 - 43 ではエッジ検出手法の一つであるCanny法の理論となる。

Canny法は、
1. ガウシアンフィルタを掛ける
2. x, y方向のSobelフィルタを掛け、それらからエッジ強度とエッジ勾配を求める
3. エッジ勾配の値から、Non-maximum suppression によりエッジの細線化を行う
4. ヒステリシスによる閾値処理を行う

以上により、画像からエッジ部分を抜き出す手法である。

ここでは、1と2の処理を実装する。

処理手順は、
1. 画像をグレースケール化する
2. ガウシアンフィルタ(5x5, s=1.4)をかける
3. x方向、y方向のsobelフィルタを掛け、画像の勾配画像fx, fyを求め、勾配強度と勾配角度を次式で求める。

```bash
勾配強度 edge = sqrt(fx^2 + fy^2)
勾配角度 angle = arctan(fy / fx)
```

4. 勾配角度を次式に沿って、量子化する。

ただし、angleはradianから角度(degree)にして、-22.5から157.5の範囲をとるように値が修正してから、以下の計算を行う。

```bash
angle = {   0  (if -22.5 < angle <= 22.5)
           45  (if 22.5 < angle <= 67.5)
           90  (if 67.5 < angle <= 112.5)
          135  (if 112.5 < angle <= 157.5)
```

ここではガウシアンフィルタとsobelはopencvの関数を使う。


```python
def get_edge_angle(fx, fy):
    edge = np.sqrt(np.power(fx, 2) + np.power(fy, 2))
    fx = np.maximum(fx, 1e-5)
    angle = np.arctan(fy / fx)
    return edge, angle


def angle_quantization(angle):
    angle = angle / np.pi * 180
    angle[angle < -22.5] = 180 + angle[angle < -22.5]
    _angle = np.zeros_like(angle, dtype=np.uint8)
    _angle[np.where(angle <= 22.5)] = 0
    _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
    _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
    _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135
    return _angle


# gaussian filtering
gaussian = cv2.GaussianBlur(img_gray, (5, 5), 1.4)
fx = cv2.Sobel(gaussian, cv2.CV_32F, 1, 0, ksize=3)
fy = cv2.Sobel(gaussian, cv2.CV_32F, 0, 1, ksize=3)

# get edge strength, angle
edge, angle = get_edge_angle(fx, fy)

# angle quantization
angle = angle_quantization(angle)

plt.subplot(1, 3, 1)
plt.title("input")
plt.imshow(img_gray, cmap="gray")
plt.subplot(1, 3, 2)
plt.title("edge")
plt.imshow(edge, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("angle")
plt.imshow(angle, cmap="gray")
plt.show()
```


    
![png](images/output_3_0.png)
    



```python
angle
```




    array([[ 0,  0,  0, ...,  0,  0,  0],
           [90, 90, 90, ..., 90, 90, 90],
           [90, 90, 90, ..., 90, 90, 90],
           ...,
           [90, 90, 90, ..., 90, 90, 90],
           [90, 90, 90, ..., 90, 90, 90],
           [ 0,  0,  0, ...,  0,  0,  0]], dtype=uint8)



# Q.42. Cannyエッジ検出 (Step.2) 細線化

ここでは3を実装する。

Q.41で求めた勾配角度から、Non-maximum suppressionを行い、エッジ線を細くする（細線化）。

Non-maximum suppression(NMS)とは非最大値以外を除去する作業の総称である。（他のタスクでもこの名前はよく出る）

ここでは、注目している箇所の勾配角度の法線方向の隣接ピクセルの３つの勾配強度を比較して、最大値ならそのまま値をいじらずに、最大値でなければ強度を0にする、

つまり、勾配強度edge(x,y)に注目している際に、勾配角度angle(x,y)によって次式のようにedge(x,y)を変更する。

```python
if angle(x,y)  = 0
 if edge(x,y), edge(x-1,y), edge(x+1,y)で edge(x,y)が最大じゃない
  then edge(x,y) = 0
if angle(x,y)  = 45
 if edge(x,y), edge(x-1,y+1), edge(x+1,y-1)で edge(x,y)が最大じゃない
  then edge(x,y) = 0
if angle(x,y)  = 90
 if edge(x,y), edge(x,y-1), edge(x,y+1)で edge(x,y)が最大じゃない
  then edge(x,y) = 0
if angle(x,y)  = 135
 if edge(x,y), edge(x-1,y-1), edge(x+1,y+1)で edge(x,y)が最大じゃない
  then edge(x,y) = 0
```


```python
def non_maximum_suppression(angle, edge):
    H, W = angle.shape
    _edge = edge.copy()

    for y in range(H):
        for x in range(W):
            if angle[y, x] == 0:
                dx1, dy1, dx2, dy2 = -1, 0, 1, 0
            elif angle[y, x] == 45:
                dx1, dy1, dx2, dy2 = -1, 1, 1, -1
            elif angle[y, x] == 90:
                dx1, dy1, dx2, dy2 = 0, -1, 0, 1
            elif angle[y, x] == 135:
                dx1, dy1, dx2, dy2 = -1, -1, 1, 1
            if x == 0:
                dx1 = max(dx1, 0)
                dx2 = max(dx2, 0)
            if x == W-1:
                dx1 = min(dx1, 0)
                dx2 = min(dx2, 0)
            if y == 0:
                dy1 = max(dy1, 0)
                dy2 = max(dy2, 0)
            if y == H-1:
                dy1 = min(dy1, 0)
                dy2 = min(dy2, 0)
            if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
                _edge[y, x] = 0
                        
    return _edge
                        
edge_nms = non_maximum_suppression(angle, edge)

plt.subplot(1, 3, 1)
plt.title("input")
plt.imshow(img_gray, cmap="gray")
plt.subplot(1, 3, 2)
plt.title("edge before nms")
plt.imshow(edge, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("edge")
plt.imshow(edge_nms, cmap="gray")
plt.show()
```


    
![png](images/output_6_0.png)
    


# Q.43. Cannyエッジ検出 (Step.3) ヒステリシス閾処理
ここでは4を実装する。これがCanny法の最後である。

ここでは、閾値により勾配強度の二値化を行うがCanny法では二つの閾値(HT: high thoresholdとLT: low threshold)を用いる。

はじめに、

- 勾配強度edge(x,y)がHT以上の場合はedge(x,y)=255
- LT以下のedge(x,y)=0
LT < edge(x,y) < HTの時、周り８ピクセルの勾配強度でHTより大きい値が存在すれば、edge(x,y)=255
ここでは、HT=50, LT=20とする。ちなみに閾値の値は結果を見ながら判断するしかない。

以上のアルゴリズムによって、Canny法が行われる。


```python
def hysterisis(edge, HT=100, LT=30):
    H, W = edge.shape

    # Histeresis threshold
    edge[edge >= HT] = 255
    edge[edge <= LT] = 0

    _edge = np.zeros((H + 2, W + 2), dtype=np.float32)
    _edge[1 : H + 1, 1 : W + 1] = edge

    ## 8 - Nearest neighbor
    nn = np.array(((1., 1., 1.), (1., 0., 1.), (1., 1., 1.)), dtype=np.float32)

    for y in range(1, H+2):
        for x in range(1, W+2):
            if _edge[y, x] < LT or _edge[y, x] > HT:
                continue
            if np.max(_edge[y-1:y+2, x-1:x+2] * nn) >= HT:
                _edge[y, x] = 255
            else:
                _edge[y, x] = 0

    edge = _edge[1:H+1, 1:W+1]

    return edge

out = hysterisis(edge_nms, 50, 20)

plt.subplot(1, 3, 1)
plt.title("input")
plt.imshow(img_gray, cmap="gray")
plt.subplot(1, 3, 2)
plt.title("canny")
plt.imshow(out, cmap="gray")
plt.show()
```


    
![png](images/output_8_0.png)
    


# Q.44. Hough変換・直線検出 (Step.1) Hough変換

Q.44 - 46 ではHough変換を用いた直線検出を行う。

Hough変換とは、座標を直交座標から極座標に変換することにより数式に沿って直線や円など一定の形状を検出する手法である。 ある直線状の点では極座標に変換すると一定のr, tにおいて交わる。 その点が検出すべき直線を表すパラメータであり、このパラメータを逆変換すると直線の方程式を求めることができる。

方法としては、

エッジ画像(ここではCannyの出力)からエッジのピクセルにおいてHough変換を行う。
Hough変換後の値のヒストグラムをとり、極大点を選ぶ。
極大点のr, tの値をHough逆変換して検出した直線のパラメータを得る。
となる。

ここでは、1のHough変換を行いヒストグラムを作成する。

アルゴリズムは、

画像の対角線の長さrmaxを求める。
エッジ箇所(x,y)において、t = 0-179で一度ずつtを変えながら、次式によりHough変換を行う。
rho = x * cos(t) + y * sin(t)
「ボーティング（投票）」 [2 x rmax, 180]のサイズの表を用意し、1で得たtable(rho, t) に1を足す。２で求めたrhoは[-rmax, rmax]の範囲を取るので、ボーティングの時には注意！
ボーティングでは、一定の箇所に集中する。

今回はtorino.jpgを用いて、ボーディングした表を図示せよ。 Cannyのパラメータは, gaussian filter(5x5, s=1.4), HT = 100, LT = 30で使用せよ。


```python
def voting(edge):
    H, W = edge.shape
    drho = 1
    dtheta = 1

    # get rho max length
    rho_max = np.ceil(np.sqrt(H ** 2 + W ** 2)).astype(int)

    # hough table
    hough = np.zeros((rho_max * 2, 180), dtype=int)

    # get index of edge
    ind = np.where(edge == 255)

    ## hough transformation
    for y, x in zip(ind[0], ind[1]):
        for theta in range(0, 180, dtheta):
            # get polar coordinat4s
            t = np.pi / 180 * theta
            rho = int(x * np.cos(t) + y * np.sin(t))
            # vote
            hough[rho + rho_max, theta] += 1

    out = hough.astype(np.uint8)
    return out

gaussian = cv2.GaussianBlur(img_thorino_gray, (5, 5), 1.4)
fx = cv2.Sobel(gaussian, cv2.CV_32F, 1, 0, ksize=3)
fy = cv2.Sobel(gaussian, cv2.CV_32F, 0, 1, ksize=3)

edge, angle = get_edge_angle(fx, fy)
angle = angle_quantization(angle)
edge_nms = non_maximum_suppression(angle, edge)
canny = hysterisis(edge_nms, 50, 20)

vote = voting(canny)

plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(img_thorino)
plt.subplot(1, 2, 2)
plt.title("vote")
plt.imshow(vote)
plt.show()
```


    
![png](images/output_10_0.png)
    


# Q.45. Hough変換・直線検出 (Step.2) NMS

ここでは2を実装する。

Q.44で得られた表では、ある一定の箇所付近に多く投票される。 ここでは、その付近の極大値を抜き出す操作を行え。

今回はボーディングが多い箇所を上位20個抜き出し、図示せよ。(C++の解答は上位30個にしてます。なんか20だと同じような結果にらなかったので、、)

NMSのアルゴリズムは、

表において、周囲8マス(8近傍)より注目ピクセルの得票数が多ければそのまま。
注目ピクセルの値が少なければ0にする。


```python
# non maximum suppression
def non_maximum_suppression_hough_lines(hough):
    rho_max, _ = hough.shape

    ## non maximum suppression
    for y in range(rho_max):
        for x in range(180):
            # get 8 nearest neighbor
            x1 = max(x-1, 0)
            x2 = min(x+2, 180)
            y1 = max(y-1, 0)
            y2 = min(y+2, rho_max-1)
            if np.max(hough[y1:y2, x1:x2]) == hough[y,x] and hough[y, x] != 0:
                pass
                #hough[y,x] = 255
            else:
                hough[y,x] = 0

    # for hough visualization
    # get top-10 x index of hough table
    ind_x = np.argsort(hough.ravel())[::-1][:20]
    # get y index
    ind_y = ind_x.copy()
    thetas = ind_x % 180
    rhos = ind_y // 180
    _hough = np.zeros_like(hough, dtype=int)
    _hough[rhos, thetas] = 255

    return _hough


vote_nms = non_maximum_suppression_hough_lines(vote)

plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(img_thorino)
plt.subplot(1, 2, 2)
plt.title("vote")
plt.imshow(vote_nms)
plt.show()
```


    
![png](images/output_12_0.png)
    


# Q.46. Hough変換・直線検出 (Step.3) Hough逆変換

ここではQ.45.で得られた極大値をHough逆変換をして直線を描画する。これで、Hough変換による直線検出が完了する。

アルゴリズムは、

極大点(r, t)を次式で逆変換する。
- y = - cos(t) / sin(t) * x + r / sin(t)
- x = - sin(t) / cos(t) * y + r / cos(t)

1の逆変換を極大点ごとにy = 0 - H-1, x = 0 - W-1 で行い、入力画像に検出した直線を描画せよ。 ただし、描画するのは赤線(R,G,B) = (255, 0, 0)とする。


```python
def inverse_hough(hough, img):
    H, W, _ = img.shape
    rho_max, _ = hough.shape

    out = img.copy()

    # get x, y index of hough table
    ind_x = np.argsort(hough.ravel())[::-1][:20]
    ind_y = ind_x.copy()
    thetas = ind_x % 180
    rhos = ind_y // 180 - rho_max / 2

    # each theta and rho
    for theta, rho in zip(thetas, rhos):
        # theta[radian] -> angle[degree]
        t = np.pi / 180. * theta

        # hough -> (x,y)
        for x in range(W):
            if np.sin(t) != 0:
                y = - (np.cos(t) / np.sin(t)) * x + (rho) / np.sin(t)
                y = int(y)
                if y >= H or y < 0:
                    continue
                out[y, x] = [255, 0, 0]
        for y in range(H):
            if np.cos(t) != 0:
                x = - (np.sin(t) / np.cos(t)) * y + (rho) / np.cos(t)
                x = int(x)
                if x >= W or x < 0:
                    continue
                out[y, x] = [255, 0, 0]

    out = out.astype(np.uint8)

    return out


out = inverse_hough(vote_nms, img_thorino)

plt.subplot(1, 2, 1)
plt.title("input")
plt.imshow(img_thorino)
plt.subplot(1, 2, 2)
plt.title("hough")
plt.imshow(out)
plt.show()
```


    
![png](images/output_14_0.png)
    


# Q.47. モルフォロジー処理(膨張)

Morphology Dilate

imori.jpgを大津の二値化したものに、モルフォロジー処理による膨張を2回行え。

モルフォロジー処理とは二値化画像の白(255)マス部分を4近傍(上下左右1マス)に膨張、または1マスだけ収縮させる処理をいう。

この膨張と収縮を何度も繰り返すことで1マスだけに存在する白マスを消したり(Q.49. オープニング処理)、本来つながってほしい白マスを結合させたりできる(Q.50. クロージング処理)。

モルフォロジー処理の膨張(Dilation)アルゴリズムは、 注目画素I(x, y)=0で、I(x, y-1), I(x-1, y), I(x+1, y), I(x, y+1)のどれか一つが255なら、I(x, y) = 255 とする。

<img src="images/morphology_erode.png" width=200>

つまり、上の処理を2回行えば2マス分膨張できることになる。

モルフォロジー処理の実装は例えば、[[0,1,0], [1,0,1], [0,1,0]] のフィルタを掛けた和が255以上なら膨張である、と考えることもできる。


```python
def morphology_erode(img, repeat=1):
    h, w = img.shape
    out = img.copy()

    # kernel
    mf = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=int)

    # each erode
    for i in range(repeat):
        tmp = np.pad(out, (1, 1), 'edge')
        # erode
        for y in range(1, h + 1):
            for x in range(1, w + 1):
                if np.sum(mf * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                    out[y-1, x-1] = 0

    return out

_, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Morphology - dilate
out = morphology_erode(otsu, repeat=2)

plt.subplot(1, 3, 1)
plt.title("input")
plt.imshow(img_orig)
plt.subplot(1, 3, 2)
plt.title("otsu binary")
plt.imshow(otsu, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("erode")
plt.imshow(out, cmap="gray")
plt.show()
```


    
![png](images/output_16_0.png)
    


# Q.48. モルフォロジー処理(収縮)

imori.jpgを大津の二値化したものに、モルフォロジー処理による収縮を2回行え。

モルフォロジー処理の収縮(Erosion)アルゴリズムは、 注目画素I(x, y)=255で、I(x, y-1), I(x-1, y), I(x+1, y), I(x, y+1)のどれか一つでも0なら、I(x, y) = 0 とする。

収縮処理は例えば、[[0,1,0], [1,0,1], [0,1,0]] のフィルタを掛けた和が255*4未満なら収縮である、と考えることもできる。

<img src="images/morphology_dilate.png" width=200>


```python
def morphology_dilate(img, repeat=1):
    h, w = img.shape

    # kernel
    mf = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=int)

    # each dilate time
    out = img.copy()
    for i in range(repeat):
        tmp = np.pad(out, (1, 1), 'edge')
        for y in range(1, h+1):
            for x in range(1, w+1):
                if np.sum(mf * tmp[y-1:y+2, x-1:x+2]) >= 255:
                    out[y-1, x-1] = 255

    return out


_, otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# Morphology - dilate
out = morphology_dilate(otsu, repeat=2)

plt.subplot(1, 3, 1)
plt.title("input")
plt.imshow(img_orig)
plt.subplot(1, 3, 2)
plt.title("otsu binary")
plt.imshow(otsu, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("dilate")
plt.imshow(out, cmap="gray")
plt.show()
```


    
![png](images/output_18_0.png)
    


# Q.49. オープニング処理
大津の二値化後に、オープニング処理(N=1)を行え。

オープニング処理とは、モルフォロジー処理の収縮をN回行った後に膨張をN回行う処理である。

<img src="images/morphology_opening.png" width=400>

オープニング処理により、一つだけ余分に存在する画素などを削除できる。


```python
time = 1
out = morphology_dilate(otsu, repeat=time)
out = morphology_erode(out, repeat=time)

plt.subplot(1, 3, 1)
plt.title("input")
plt.imshow(img_orig)
plt.subplot(1, 3, 2)
plt.title("otsu binary")
plt.imshow(otsu, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("opening")
plt.imshow(out, cmap="gray")
plt.show()
```


    
![png](images/output_20_0.png)
    


# Q.50. クロージング処理

Canny検出した後に、クロージング処理(N=1)を行え。

クロージング処理とは、モルフォロジー処理の膨張をN回行った後に収縮をN回行う処理である。

<img src="images/morphology_closing.png" width=400>

クロージング処理により、途中で途切れた画素を結合することができる。


```python
time = 1
out = morphology_erode(otsu, repeat=time)
out = morphology_dilate(out, repeat=time)


plt.subplot(1, 3, 1)
plt.title("input")
plt.imshow(img_orig)
plt.subplot(1, 3, 2)
plt.title("otsu binary")
plt.imshow(otsu, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("closing")
plt.imshow(out, cmap="gray")
plt.show()
```


    
![png](images/output_22_0.png)
    

