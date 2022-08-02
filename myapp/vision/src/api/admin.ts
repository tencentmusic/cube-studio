import ajax from './ajax';

const baseUrl = 'http://11.187.53.46:8080/api'


const query_abtests = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_abtests`,
        data:  data
    });
}
const query_edge_factorys = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_edge_factorys`,
        data:  data
    });
}
const delete_edge_factory = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/delete_edge_factory`,
        data:  data
    });
}
const query_structs = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_structs`,
        data:  data
    });
}
const delete_struct = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/delete_struct`,
        data:  data
    });
}
const query_associated_nodes = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_associated_nodes`,
        data:  data
    });
}
const mod_struct = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/mod_struct`,
        data:  data
    });
}
const add_struct = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/add_struct`,
        data:  data
    });
}
const query_scenes = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_scenes`,
        data:  data
    });
}
const query_associated_graphs = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_associated_graphs`,
        data:  data
    });
}
const mod_scene = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/mod_scene`,
        data:  data
    });
}
const add_scene = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/add_scene`,
        data:  data
    });
}
const query_graph_templates = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_graph_templates`,
        data:  data
    });
}
const query_components = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_components`,
        data:  data
    });
}
const delete_component = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/delete_component`,
        data:  data
    });
}
const query_component_historys = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_component_historys`,
        data:  data
    });
}
const get_scene_names = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/get_scene_names`,
        data:  data
    });
}
const get_scene_available_graphs = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/get_scene_available_graphs`,
        data:  data
    });
}
const mod_abtest = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/mod_abtest`,
        data:  data
    });
}
const add_abtest = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/add_abtest`,
        data:  data
    });
}
const query_graphs = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_graphs`,
        data:  data
    });
}
const get_graph_template_names = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/get_graph_template_names`,
        data:  data
    });
}
const batch_query_components = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/batch_query_components`,
        data:  data
    });
}

const query_graph_historys = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_graph_historys`,
        data:  data
    });
}
const query_template_available_node_factorys = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_template_available_node_factorys`,
        data:  data
    });
}
const get_edge_factory_names = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/get_edge_factory_names`,
        data:  data
    });
}
const get_node_factory_names = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/get_node_factory_names`,
        data:  data
    });
}
const query_node_factorys = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/query_node_factorys`,
        data:  data
    });
}
const mod_graph = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/mod_graph`,
        data:  data
    });
}
const add_graph = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/add_graph`,
        data:  data
    });
}
const get_config_prototype = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/get_config_prototype`,
        data:  data
    });
}
const rollback_graph = (data: any): Promise<any> => {
    return ajax.post({
        url: `${baseUrl}/rollback_graph`,
        data:  data
    });
}



export default {
    query_abtests,
    query_edge_factorys,
    delete_edge_factory,
    query_structs,
    delete_struct,
    query_associated_nodes,
    mod_struct,
    add_struct,
    query_scenes,
    query_associated_graphs,
    add_scene,
    mod_scene,
    query_graph_templates,
    query_components,
    delete_component,
    query_component_historys,
    get_scene_names,
    get_scene_available_graphs,
    mod_abtest,
    add_abtest,
    query_graphs,
    get_graph_template_names,
    batch_query_components,
    query_graph_historys,
    query_template_available_node_factorys,
    get_edge_factory_names,
    get_node_factory_names,
    query_node_factorys,
    mod_graph,
    add_graph,
    get_config_prototype,
    rollback_graph
}