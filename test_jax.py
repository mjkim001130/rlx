# test_jax_flax.py
import os
import jax
import jax.numpy as jnp
from jax import jit, grad, random
from flax import linen as nn
from flax.core import FrozenDict


def banner(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def check_jax():
    banner("[1] JAX 환경 확인")
    print(f"JAX version   : {jax.__version__}")
    try:
        import jaxlib

        print(f"jaxlib version: {jaxlib.__version__}")
    except Exception:
        print("jaxlib version: (확인 실패, 생략)")

    # 사용 가능한 장치 출력
    devices = jax.devices()
    print("Available devices:")
    for i, d in enumerate(devices):
        print(f"  [{i}] {d}")

    # 기본 백엔드/선호 장치
    print(f"default backend: {jax.default_backend()}")

    # 간단 연산 + JIT + GRAD
    @jit
    def f(x, y):
        return jnp.sin(x) * jnp.cos(y) + x * y

    x = jnp.array(1.2345, dtype=jnp.float32)
    y = jnp.array(2.3456, dtype=jnp.float32)
    out = f(x, y)
    print(
        f"JIT func output f(x,y): {out} (device: {out.device_buffer.device() if hasattr(out, 'device_buffer') else 'n/a'})"
    )

    # grad 검사 (x에 대한 도함수)
    df_dx = grad(lambda a: f(a, y))(x)
    print(f"d/dx f(x,y) at x={float(x)}: {float(df_dx)}")


def check_flax():
    banner("[2] Flax 모델/학습 한 스텝 확인")

    # 간단 MLP 정의
    class MLP(nn.Module):
        hidden: int = 32

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.hidden)(x)
            x = nn.relu(x)
            x = nn.Dense(1)(x)
            return x

    key = random.PRNGKey(0)

    # 가짜 데이터 (회귀)
    n, d = 128, 4
    key_x, key_w, key_noise = random.split(key, 3)
    X = random.normal(key_x, (n, d))
    true_w = random.normal(key_w, (d, 1))
    y = X @ true_w + 0.1 * random.normal(key_noise, (n, 1))  # y = Xw + noise

    model = MLP(hidden=32)

    # 파라미터 초기화
    params: FrozenDict = model.init(random.PRNGKey(1), jnp.zeros((1, d)))
    print("Init param tree:", jax.tree_util.tree_map(lambda a: a.shape, params))

    # 손실 함수(MSE)
    def mse_loss(p, xb, yb):
        pred = model.apply(p, xb)
        return jnp.mean((pred - yb) ** 2)

    # 파라미터 그라디언트
    loss_val = mse_loss(params, X, y)
    grads = grad(mse_loss)(params, X, y)
    print(f"Initial loss: {float(loss_val):.6f}")

    # SGD 한 스텝 (학습률 1e-2)
    lr = 1e-2
    params_updated = jax.tree_util.tree_map(lambda w, g: w - lr * g, params, grads)
    loss_after = mse_loss(params_updated, X, y)
    print(f"Loss after 1 step SGD: {float(loss_after):.6f}")

    # 장치 확인을 위해 임의 텐서를 올려보고 device 출력
    sample = jnp.ones((2, 2))
    dev = sample.device_buffer.device() if hasattr(sample, "device_buffer") else "n/a"
    print(f"Flax forward ran; sample tensor device: {dev}")


if __name__ == "__main__":
    check_jax()
    check_flax()
    banner("✅ 테스트 완료: JAX/Flax 기본 동작 OK (오류 없으면 정상)")
