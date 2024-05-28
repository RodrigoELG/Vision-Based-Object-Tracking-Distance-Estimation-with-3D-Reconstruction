import open3d as o3d

def load_and_visualize_ply(file_path):
    # Cargar el archivo .ply
    pcd = o3d.io.read_point_cloud(file_path)
    
    # Verificar si el archivo se ha cargado correctamente
    if pcd.is_empty():
        print(f"Error: No se pudo cargar el archivo {file_path}")
        return

    # Visualizar el modelo
    o3d.visualization.draw_geometries([pcd], window_name="Visualizador Open3D",
                                      width=800, height=600, left=50, top=50,
                                      point_show_normal=False, mesh_show_wireframe=False, mesh_show_back_face=False)

if __name__ == "__main__":
    # Ruta al archivo .ply
    file_path = "object_3d_model.ply"
    
    # Cargar y visualizar el archivo .ply
    load_and_visualize_ply(file_path)

#stereovision opencv
#ORB
#Colmap
#ejecutable meshroom 