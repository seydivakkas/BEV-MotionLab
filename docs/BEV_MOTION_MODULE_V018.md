# BEV-MotionLab — Matematiksel çekirdek ve durum denetimi v0.1.8

**Durum:** Sentetik, Windows-first Python kütüphane artımı. **Backend v1 HOLD.** Yeni API sayısal kalite denetiminden geçse bile gerçek hız doğruluğu, kamera/lens hata bütçesi ve bağımsız saha kabulü **kanıtlanmış değildir**.

## Özellik haritası

| Gereksinim | Kod | Bağımsız sınır / garanti |
|---|---|---|
| Normalize DLT (piksel → metre) | `geometry.fit_dlt` + `motion_module.compute_homography_dlt` | 4+ yayılmış, dejenere olmayan eşleşme, ters yönde birim hatası yok |
| Piksel RANSAC | `geometry.fit_ransac` + `BEVMotionModule.from_correspondences(ransac=True)` | RANSAC eşikleri piksel cinsinden, inlier fit/holdout ayrılır |
| Jacobian `J` | `covariance_projection.project_pixel_covariance_to_bev` | `P_XY=JP_uv Jᵀ` analitik ve sayısal birinci dereceden |
| %95 elips | `motion_module.extract_error_ellipse_parameters` | `χ²(2,95%)=5.991...`, yarı büyük/küçük eksen metre |
| Hız ve ivme | `motion_module.calculate_physical_speed`, `BEVMotionModule.process_sample` | monoton PTS/saat damgası, d = metres, dt = saniye, koşullu belirsizlik |
| Ufuk/ters yön ve ROI | `BEVMotionModule.process_sample` | kalibre edilmiş convex hull dışında, ufka yakın ya da ters işaretli `w`: `UNCERTAIN_PROJECTION` |
| Düzlem dışı temas | `contact_is_ground` zorunlu + `height_risk`, `interval_validity` | zemin temasını belirsiz bildiren gözleme hız yok; bbox alt-orta noktası otomatik zemin kabul edilmez |
| Kamera/lens provenance | `from_correspondences`, `quality.assess_calibration`, mevcut `camera3d.py` / `pose_audit.py` | görüntü **önceden undistort edilmiş**, kamera ve lens ID zorunlu; bağımsız 4+ holdout |
| Kalman çapraz-zaman kovaryansı | `project_cross_frame_covariance_to_bev` | `C_XY(0,1)=J0 C_uv(0,1) J1ᵀ`; bilinmiyorsa hız güven aralığı yalnız koşullu yaklaşım |
| Sürümlü local JSON API | `api.dispatch` `project_tracks`, `loopback` | eski `calibrate`, `project_speed` işlemleri değiştirilmedi; keyfi dosya yolu kabul edilmez |
| Dosya çıktısı ve bütünlük | `motion_cli`, `examples/motion_*` | hesap tamamen başarılı olmadan çıktı klasörü açılmaz; SHA-256 manifesti ve JSONL |

## Hata ve belirsizlik sınırları

- `(u,v)` noktası için homografi **piksel → dünya metre** yönündedir:
  `X = (h11*u + h12*v + h13)/w`, `Y=(h21*u+h22*v+h23)/w`, `w=h31*u+h32*v+h33`.
- `J = [[(h11-X*h31)/w,(h12-X*h32)/w],[(h21-Y*h31)/w,(h22-Y*h32)/w]]`.
- `P_XY=J P_uv Jᵀ` yalnız **piksel-gürültüsü belirsizliği** için birinci dereceden yaklaşımdır; ortak homografi/lens/poz/survey hataları bu matriste bulunmaz.
- Hız `||P1-P0||/(t1-t0)` ve `×3.6 km/sa`. Delta metoduyla hız sigma hesaplanır. Sıfır yer değiştirmede norm türevi tanımsızdır: `ZERO_DISPLACEMENT_DIRECTION_UNDEFINED`; uydurma bir sigma raporlanmaz.
- Kovaryans eşleştirmesi `Pdelta=P0+P1-C01-C01ᵀ`. `C01` yoksa iki örnek bağımsız varsayılır ve `VALID_CONDITIONAL` çıktısı bu **varsayımı açıkça taşır**. Bu gerçek %95 hata-kapsama garantisi değildir.
- İvme, son iki hız vektörünün sayısal türevidir; `acceleration_confidence=NOT_ESTABLISHED`, kaynak-türevi korelasyonu nedeniyle ayrı bağımsız güven aralığı yoktur.
- Bounding-box alt merkezi kendiliğinden temas noktası değildir. `contact_is_ground=True` girdisi *kaynak tarafından doğrulanmış/ilan edilmiş* olmalıdır; aksi hâlde hiçbir hız türetilmez.
- Kamera değişirse, iz kimliği koparsa veya örtülme oluşursa önceki örnek yeniden birleştirilmez. Sonraki geçerli örnek yalnız yeni bir hız segmenti başlatır.
- `pixel_roi` tek başına geçerli bölge değildir; hem kullanıcı ROI'si hem kalibrasyon noktalarının konveks kılıfı uygulanır. ROI ve kılıfın dışına **ekstrapolasyon yoktur**.
- Mevcut `height_risk.py`, `camera3d.py`, `pose_audit.py`, `interval_validity.py` kontrolleri bu yeni modülden bağımsızdır; bu sürüm tek başına bunların tamamı için fiziksel garantiler vermez.

## Başlangıç ve örnek

```powershell
py -m pip install -e .
bev-motion-run --calibration examples/motion_calibration.json --input examples/motion_samples.jsonl --out outputs/demo
# veya:
py -m bevmotionlab.motion_cli --calibration examples/motion_calibration.json --input examples/motion_samples.jsonl --out outputs/demo
```

Dosyalar: `motion_report.json`, `motion_records.jsonl`, `manifest.json`. `--out` mevcut olmamalı. Kaynaklar sentetiktir; varsayılan olarak görüntü indirme/yükleme veya ücretli servis yoktur.

### Python

```python
from bevmotionlab.motion_module import BEVMotionModule
import json
c=json.load(open('examples/motion_calibration.json',encoding='utf8'))
m=BEVMotionModule.from_correspondences(**c,max_major_axis_m=3.0)
record={"track_id":"T1", "timestamp_s":0.1, "u":50.0, "v":50.0,
        "cov_uv":[[0.04,0],[0,0.04]],"contact_is_ground":True,
        "segment":"s0", "camera_id":c["camera_id"],
        "lens_calibration_id":c["lens_calibration_id"]}
print(m.process_sample(record))  # INITIALIZED; ilk örnekte hız yok
```

### Yerel JSON API

Gövde `{ "schema_version":"1.0", "operation":"project_tracks", "input": { "calibration": <yukarıdaki calibration JSON>, "samples": [<yukarıdaki track kayıtları>], "limits": {"max_major_axis_m": 3.0} } }`.

`/v1/execute` üzerinde eski bearer-token'lı loopback sunucusu `api.dispatch` işlemini kullanır. Sıradan tarayıcı `Origin` başlıklı isteklerle yetkilendirilmez. UI entegrasyonu için ayrıca güvenli ve oturum kontrollü bir adaptör geliştirilmelidir.

## Test ve yeni PR kabulü

```powershell
py -m unittest discover -s tests -v
```

- Matematik: noiseless DLT, Jacobian, elips, C01'nin yönlü projeksiyonu.
- Fail-closed: bozuk/PSD olmayan kovaryans, eksik lens, geometrik holdout uyumsuzluğu, yanlış lens/kamera, ufuk, temassız veya örtülmüş nesne, zaman boşluğu, hız sapması.
- Kaynak: birden fazla iz, segment, ardışık iki hız örneği, hız ve ivme.
- Paket: Windows/Linux Python 3.11/3.12 CI, başlıktan sona sentetik JSONL + SHA-256 doğrulama.

**Açık kabul:** Sistematik homografi/lens yanlılığı, kamera pozu drift stresleri, Kalman'dan çapraz-zaman bloklarının bütünleşik akışı, yanlış-zemin-temas verisi üzerinde bağımsız temassız/hata sınırı stresleri, kaynak tüketimi, Windows imzalı/offline paketleme ve *en son* bağımsız gerçek kamera verisi. **Backend v1 HOLD**. Merge yok.
