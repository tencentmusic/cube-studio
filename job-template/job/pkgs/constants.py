
KF_UI_META_FILE = "/mlpipeline-ui-metadata.json"
JOB_DEF_NAMESPACE = "pipeline"
WORKER_DEF_RESOURCE_LIMITS = {
    "limits": {
        "memory": "16G",
        "cpu": "16"
    }
}
DEF_IMAGE_PULL_SECRETS = []


class ComponentOutput(object):
    DATA_FETCH_OUTPUT = ".data_fetch_output"
    DATA_TRANSFORM_OUTPUT = ".data_transfrom_output"
    MODEL_TRAIN_OUTPUT = ".model_train_output"
    MODEL_EVALUATION_OUTPUT = ".model_evaluation_output"
    MODEL_DEPLOY_OUTPUT = ".model_deploy_output"
    HDFS_DATA_IMPORT_OUTPUT = ".hdfs_data_import_output"


class NodeAffnity(object):
    AVAILABLE = "available"
    PREF_GPU = "pref_gpu"
    ONLY_GPU = "only_gpu"
    PREF_CPU = "pref_cpu"
    ONLY_CPU = "only_cpu"


class PodAffnity(object):
    SPREAD = "spread"
    CONCENT = "concent"


class PipelineParam(object):
    PACK_PATH_PAT = '\\${(PROJ_PATH|PACK_PATH)}\\$'
    RUN_PATH_PAT = '\\${(RUN_PATH|DATA_PATH)}\\$'
    DATE_PAT = '\\${DATE((-|\\+)(\\d+)(d|w|h|m|s|y|M))?(:(.+))?}\\$'
    ONLINE_MODEL = '${ONLINE_MODEL}$'


class ModelStatus(object):
    ONLINE = 'online'
    OFFLINE = 'offline'


class AWFUserFunc(object):
    CRETAE_MODEL = 'awf_create_model_fn'
    CREATE_TRAIN_DATASET = 'awf_create_train_dataset_fn'
    CREATE_VALIDATION_DATASET = 'awf_create_val_dataset_fn'
    CREATE_TEST_DATASET = 'awf_create_test_dataset_fn'
    CREATE_PREDICTION_DATASET = 'awf_create_predict_dataset_fn'
    CREATE_MODEL_TO_SAVE = 'awf_model_to_save_fn'
    LOAD_MODEL = 'awf_load_model_fn'
    GROUP_TRAINABLE_VARS = "awf_group_trainable_vars_fn"


class DatasetType(object):
    TRAIN = 'train'
    TEST = 'test'
    VALIDATION = 'val'
    PREDICTION = 'pred'


class DistributionType(object):
    PS = 'ps'
    SINGLE_WORKER = 'single_worker'
    MULTI_WORKER = 'multi_worker'
    NONE = 'None'


class ComputeResource(object):
    P_GPU = "nvidia.com/gpu"
    V_GPU_CORE = "tencent.com/vcuda-core"
    V_GPU_MEM = "tencent.com/vcuda-memory"
    V_GPU_MEM_UNIT = 256 * (2 ** 20)

    class ClusterEnv(object):
        TKE = 'TENCENT'
