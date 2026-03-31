import streamlit as st
import torch
import matplotlib.pyplot as plt
from sklearn import svm

st.set_page_config(page_title="SVM Expert Lab", layout="wide")

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; }
    .stMarkdown p { font-size: 15px; line-height: 1.6; }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 SVM Deep Analytics: Phân tích Tác động Tham số")
st.markdown("---")

problems = {
    'AND': (torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]]), torch.tensor([0, 0, 0, 1])),
    'OR':  (torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]]), torch.tensor([0, 1, 1, 1])),
    'XOR': (torch.tensor([[0.,0.], [1.,1.], [1.,0.], [0.,1.]]), torch.tensor([0, 0, 1, 1])),
    'NOT': (torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]]), torch.tensor([1, 1, 0, 0]))
}

def get_best_params(kernel, prob):
    if kernel == 'linear': return (1.0, 0.0)
    if kernel == 'rbf': return (5.0, 0.0) if prob == 'XOR' else (1.0, 0.0)
    if kernel == 'poly': return (2.0, 2.0) if prob == 'XOR' else (1.0, 1.0)
    if kernel == 'sigmoid': return (10.0, -5.0) if prob == 'XOR' else (2.0, 0.0)
    return (1.0, 0.0)

with st.sidebar:
    st.header("⚙️ Thiết lập Mô hình")
    prob_name = st.selectbox("1. Chọn Bài toán:", list(problems.keys()))
    kernel_type = st.radio("2. Chọn Kernel:", ('linear', 'rbf', 'poly', 'sigmoid'))
    
    default_g, default_r = get_best_params(kernel_type, prob_name)
    
    st.markdown("---")
    st.subheader("🎯 Tinh chỉnh Hình học")
    gamma = st.slider(r"Hệ số Gamma ($\gamma$):", 0.01, 50.0, default_g, key=f"g_{prob_name}_{kernel_type}")
    coef0 = st.slider(r"Hệ số Coef0 ($r$):", -10.0, 10.0, default_r, key=f"r_{prob_name}_{kernel_type}")

X_curr, y_curr = problems[prob_name]
clf = svm.SVC(kernel=kernel_type, gamma=gamma, coef0=coef0, C=1e6)
clf.fit(X_curr.numpy(), y_curr.numpy())

h = 0.03
xx_range = torch.arange(-0.5, 1.5, h)
yy_range = torch.arange(-0.5, 1.5, h)
xx, yy = torch.meshgrid(xx_range, yy_range, indexing='xy')
grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
Z = clf.decision_function(grid.numpy()).reshape(xx.shape)

col_plot, col_analysis = st.columns([5, 4])

with col_plot:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.contourf(xx.numpy(), yy.numpy(), Z, levels=50, cmap=plt.cm.RdBu, alpha=0.3)
    ax.contour(xx.numpy(), yy.numpy(), Z, levels=[-1, 0, 1], colors='black', linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
    
    if len(clf.support_vectors_) > 0:
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=250,
                   facecolors='none', edgecolors='gold', linewidths=2.5, label='Support Vectors', zorder=10)
    
    X_np = X_curr.numpy()
    y_np = y_curr.numpy()
    ax.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1], c='red', s=150, edgecolors='k', zorder=15)
    ax.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1], c='blue', s=150, marker='s', edgecolors='k', zorder=15)
    ax.set_title(f"Mặt phẳng quyết định: {kernel_type.upper()} trên {prob_name}", fontsize=14)
    st.pyplot(fig)

with col_analysis:
    with st.expander("📌 ĐỊNH NGHĨA: GAMMA LÀ GÌ? COEF0 LÀ GÌ?", expanded=False):
        st.markdown(r"""
        * **Gamma ($\gamma$):** Quyết định *tầm ảnh hưởng* của một điểm dữ liệu duy nhất. Giá trị lớn có nghĩa là ảnh hưởng gần (chỉ các điểm sát nhau mới liên quan), giá trị nhỏ có nghĩa là ảnh hưởng xa (toàn cục).
        * **Coef0 ($r$):** Là hệ số tự do (Bias) được cộng vào trong công thức của Kernel phi tuyến. Nó giúp dịch chuyển đường biên hoặc cân bằng sự ảnh hưởng giữa các thành phần bậc cao và bậc thấp trong hàm.
        """)
        
    st.subheader(f"📖 Tác động tham số: {kernel_type.upper()} + {prob_name}")
    
    if kernel_type == 'linear':
        st.latex(r"K(x, y) = x^T y")
        st.write(f"**Trạng thái với {prob_name}:** {'Hoàn hảo (Dữ liệu tuyến tính).' if prob_name != 'XOR' else 'Thất bại (Không thể dùng 1 đường thẳng cắt XOR).'}")
        with st.expander("📌 BÓC TÁCH TOÁN HỌC: LINEAR KERNEL", expanded=False):
            st.markdown(r"""
            Hàm Linear thực hiện phép chiếu tuyến tính thuần túy trong không gian gốc.
            
            **🎯 Tác động tham số:**
            * Cả **Gamma ($\gamma$)** và **Coef0 ($r$)** đều **không có tác dụng** đối với Linear Kernel, vì công thức $x^T y$ không chứa hai tham số này.
            """)

    elif kernel_type == 'poly':
        st.latex(r"K(x, y) = (\gamma x^T y + r)^d")
        st.write(f"**Trạng thái với {prob_name}:** {'Tạo đường biên cong uốn lượn.' if prob_name != 'XOR' else 'Tạo đường Hyperbol lách qua 2 điểm chéo nhau.'}")
        with st.expander("📌 BÓC TÁCH TOÁN HỌC: POLY KERNEL", expanded=False):
            st.markdown(r"""
            Để thấy rõ tại sao $\gamma$ lại bóp méo đường biên, chúng ta hãy khai triển nhị thức công thức trên với bậc $d = 2$.
            
            Ta có:
            $$K(x, y) = (\gamma x^T y + r)^2 = \gamma^2(x^T y)^2 + 2\gamma r(x^T y) + r^2$$
            
            **A. Khi Tăng $\gamma$ (Đường biên sắc nét và uốn gắt)**
            * **Khảo sát Đại số:** Khi $\gamma$ tăng lên (ví dụ từ $1$ lên $10$), hệ số $\gamma^2$ sẽ bùng nổ theo cấp số nhân (từ $1$ lên $100$).
            * **Hệ quả:** Thành phần phi tuyến bậc cao $\gamma^2(x^T y)^2$ sẽ hoàn toàn thống trị phương trình. Nó biến không gian đặc trưng thành một mặt cong (Paraboloid/Hyperboloid) cực kỳ dốc đứng.
            * **Tác động lên đồ thị:** Giống như đồ thị hàm số $y = ax^2$, khi hệ số $a$ (tương đương $\gamma^2$) khổng lồ, parabol sẽ bị ép hẹp lại và hai nhánh dựng đứng lên. Một sự thay đổi cực nhỏ của tọa độ $x$ cũng làm giá trị Kernel thay đổi dữ dội $\rightarrow$ Đường biên bị bẻ gập đột ngột ôm sát Support Vectors.

            **B. Khi Giảm $\gamma$ (Bề mặt thoai thoải, đường biên mượt mà)**
            * **Khảo sát Đại số:** Khi $\gamma$ tiến sát về $0$ (ví dụ $\gamma = 0.1$), hệ số $\gamma^2$ sẽ teo nhỏ lại cực kỳ nhanh ($0.1^2 = 0.01$).
            * **Hệ quả:** Lúc này, "kẻ thống trị" phi tuyến bị tước mất quyền lực. Mặt cong 3D không còn dốc đứng nữa mà xẹp xuống, trở nên thoai thoải như một ngọn đồi thấp.
            * **Tác động lên đồ thị:** Khoảng không gian chuyển giao từ phe Đỏ sang phe Xanh được nới rộng ra. Đạo hàm (độ dốc) của hàm mặt phẳng rất nhỏ, khiến thuật toán SVM đủ không gian để vạch ra một đường ranh giới (Decision Boundary) duỗi dài, mềm mại hơn.
            
            **🎯 Tác động của Coef0 ($r$):**
            * Đóng vai trò cân bằng giữa phần bậc cao và bậc thấp. Tăng $r$ giúp đẩy đa thức xa gốc tọa độ, linh hoạt lách qua dữ liệu (đặc biệt quan trọng với XOR). Giảm $r$ về 0 sẽ làm mất thành phần tuyến tính, khiến mô hình bị gò bó.
            """)

    elif kernel_type == 'rbf':
        st.latex(r"K(x, y) = \exp(-\gamma \|x - y\|^2)")
        st.write(f"**Trạng thái với {prob_name}:** Tạo các 'hòn đảo' (vùng ảnh hưởng) bao quanh các điểm dữ liệu.")
        with st.expander("📌 BÓC TÁCH TOÁN HỌC: RBF KERNEL", expanded=False):
            st.markdown(r"""
            **1. Bí mật "Vô hạn chiều" (Khai triển chuỗi Taylor)**
            Bí mật của sự vô hạn nằm trọn vẹn bên trong hàm số mũ của công thức RBF.
            
            * **Bước 1: Bóc tách công thức khoảng cách**
            $\|x - y\|^2 = \|x\|^2 - 2x^T y + \|y\|^2$. Thay ngược lại vào công thức RBF, ta có thành phần cốt lõi: $\exp(2\gamma x^T y)$.
            
            * **Bước 2: Kích hoạt vô hạn chiều bằng chuỗi Taylor**
            Bất kỳ hàm số mũ $e^z$ nào cũng có thể được biểu diễn dưới dạng chuỗi Maclaurin:
            $$e^z = \sum_{n=0}^{\infty} \frac{z^n}{n!} = 1 + \frac{z}{1!} + \frac{z^2}{2!} + \dots + \frac{z^\infty}{\infty!}$$
            
            * **Kết luận Đại số:** RBF chính là một tổ hợp tuyến tính của TẤT CẢ các Polynomial Kernel từ bậc 0 cho đến bậc vô cực! Nó ánh xạ điểm dữ liệu ban đầu thành một vector chứa vô số các chiều tọa độ.

            ---
            **2. Tác động của tham số: "Bán kính Ảnh hưởng"**
            * **Khi Tăng Gamma ($\gamma$):** Bán kính ảnh hưởng hẹp, đường biên giới bóp nghẹt thành các hình tròn nhỏ bó sát vào Support Vectors $\rightarrow$ Dễ Overfitting.
            * **Khi Giảm Gamma ($\gamma$):** Bán kính ảnh hưởng rộng, các vùng màu hòa quyện, đường biên trở nên thẳng và mượt hơn $\rightarrow$ Tổng quát hóa tốt hơn.

            **🎯 Tác động của Coef0 ($r$):**
            * **Không có tác dụng.** Hàm RBF thuần túy đo khoảng cách và phân rã theo hàm mũ, không chứa hệ số tự do $r$.
            """)

    elif kernel_type == 'sigmoid':
        st.latex(r"K(x, y) = \tanh(\gamma x^T y + r)")
        st.write(f"**Trạng thái với {prob_name}:** {'Tạo dải phân cách hình chữ S mềm.' if prob_name != 'XOR' else 'Chật vật với XOR do bị bão hòa (chỉ cắt được 1 góc).'}")
        with st.expander("📌 BÓC TÁCH TOÁN HỌC: SIGMOID KERNEL", expanded=False):
            st.markdown(r"""
            **1. Khi $\gamma$ Giảm (Tiến về 0): Sự thoái hóa Tuyến tính**
            Xung quanh số 0, khai triển Maclaurin cho thấy $\tanh(u) \approx u$.
            $\tanh(\gamma x^T y + r) \approx \gamma x^T y + r$. Bản chất hình học bị thoái hóa về đường thẳng tắp y hệt Linear Kernel.
            
            **2. Khi $\gamma$ Tăng (Rất lớn): Sự gãy gập và bão hòa**
            Đạo hàm của hàm $\tanh$ theo biến $z$ là: $K' = \gamma \cdot (1 - \tanh^2)$.
            Khi $\gamma$ khổng lồ, độ dốc tại ranh giới tiến đến vô cực, biến hàm $\tanh$ thành một **Hàm bước nhảy (Step Function)**.
            Điều này ép không gian vector phải bẻ gập đột ngột thành các nếp gấp (folds). Vì vi phạm định lý Mercer, sự dốc đứng này sinh ra các nghiệm nhiễu loạn, đứt gãy.
            
            **🎯 Tác động của Coef0 ($r$):**
            * Đóng vai trò là ngưỡng kích hoạt (Bias). Nó sẽ dịch chuyển toàn bộ dải phân cách (làn sóng chữ S) chạy sang trái hoặc sang phải của biểu đồ.
            """)