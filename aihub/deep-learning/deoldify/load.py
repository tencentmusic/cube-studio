
from deoldify.visualize import *

# 加载模型
plt.style.use('dark_background')
torch.backends.cudnn.benchmark = True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
self.colorizer = get_image_colorizer(root_folder=Path('/DeOldify'),artistic=True)
