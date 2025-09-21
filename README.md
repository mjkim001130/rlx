## Installation

```bash
conda create -n rlx python=3.9 -y
conda activate rlx

cd rlx
pip install -r requirements.txt
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
```

Next, install D4RL:
```bash
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
pip install numpy==1.23.0
```

---

### Error

If you got this error:
```bash
ERROR: GLEW initalization error: Missing GL version
```

then:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

