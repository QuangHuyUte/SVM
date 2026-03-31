import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from matplotlib.backends.backend_pdf import PdfPages

# 1. Tạo dữ liệu XOR
X = np.c_[(0, 0), (1, 1), (1, 0), (0, 1)].T
Y = [0] * 2 + [1] * 2

# 2. Khởi tạo một hình duy nhất với kích thước lớn (chiều ngang 15, chiều cao 5)
fig = plt.figure(figsize=(15, 5))
kernels = ('sigmoid', 'poly', 'rbf')

# Lưu vào 1 file PDF duy nhất chứa cả 3 biểu đồ
with PdfPages('svm_comparison.pdf') as pdf:
    for i, kernel in enumerate(kernels):
        clf = svm.SVC(kernel=kernel, gamma=4, coef0=0)
        clf.fit(X, Y)

        # Tạo ô lưới (Subplot) thứ i+1 trong hàng ngang 3 ô
        plt.subplot(1, 3, i + 1)

        # Vẽ các Support Vectors
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                    facecolors='none', edgecolors='k', zorder=10, label='Support Vectors')
        
        # Vẽ điểm dữ liệu thực tế
        plt.plot(X[:2, 0], X[:2, 1], 'ro', markersize=8)
        plt.plot(X[2:, 0], X[2:, 1], 'bs', markersize=8)

        # Vẽ đường biên quyết định
        x_min, x_max = -1, 2
        y_min, y_max = -1, 2
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)

        # Tô màu vùng phân loại
        plt.contourf(XX, YY, np.sign(Z), 200, cmap='jet', alpha=0.2)
        # Vẽ đường biên (nét liền là 0, nét đứt là lề margin)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-0.5, 0, 0.5])

        plt.title(f"Kernel: {kernel.upper()}", fontsize=14)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())

    plt.tight_layout() # Tự động căn chỉnh để các biểu đồ không đè lên nhau
    pdf.savefig(fig)   # Lưu cả hình lớn vào PDF
    plt.show()