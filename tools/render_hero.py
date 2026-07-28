"""Render the README hero: the machine cut open, path-traced.

Builds a Cycles scene from the mesh manifest and the page's geometry bundle: copper
non-planar and steel planar winding packs, the plasma vessel opened over a wedge with
the heat shield, baffles and divertor inside it, the plasma surface as a translucent
glowing shell, the traced field lines as emissive filaments through it, and the strike
points on the targets.

    blender -b -P tools/render_hero.py            # full frame
    HERO_PREVIEW=1 blender -b -P tools/render_hero.py   # quarter samples for framing
"""

import base64
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = Path.cwd()
MESHES = ROOT / "results/magnetics/w7x_machine_meshes.json"
BUNDLE = ROOT / "results/magnetics/w7x_geometry.json"
OUT = ROOT / "docs/w7x_twin_hero.jpg"

PREVIEW = bool(os.environ.get("HERO_PREVIEW"))
#: The wedges removed so the camera sees into the machine: the coils open over a
#: narrower span than the vessel, so the vessel's cut rim recesses behind the coil cut.
COIL_WEDGE = (-45.0, 15.0)
VESSEL_WEDGE = (-58.0, 44.0)
PLASMA_WEDGE = (-52.0, 38.0)

#: name -> (base colour, metallic, roughness, smooth shading, wedge)
SURFACES = {
    "non_planar_coils": ((0.720, 0.430, 0.180), 1.0, 0.33, True, COIL_WEDGE),
    "planar_coils": ((0.330, 0.460, 0.620), 1.0, 0.38, True, COIL_WEDGE),
    "plasma_vessel": ((0.540, 0.550, 0.570), 1.0, 0.46, True, VESSEL_WEDGE),
    "heat_shield": ((0.058, 0.062, 0.068), 0.2, 0.62, False, VESSEL_WEDGE),
    "baffle": ((0.085, 0.090, 0.098), 0.2, 0.58, False, VESSEL_WEDGE),
    "divertor": ((0.050, 0.052, 0.058), 0.3, 0.48, False, VESSEL_WEDGE),
}


def decode_block(block, key_positions="positions_b64", key_indices="indices_b64"):
    vertices = np.frombuffer(
        base64.b64decode(block[key_positions]), dtype=np.float32
    ).reshape(-1, 3).astype(np.float64)
    triangles = np.frombuffer(
        base64.b64decode(block[key_indices]), dtype=np.uint32
    ).reshape(-1, 3).astype(np.int64)
    return vertices, triangles


def decode_array(entry):
    return np.frombuffer(
        base64.b64decode(entry["data"]), dtype=entry["dtype"]
    ).reshape(entry["shape"]).astype(np.float64)


def replicate(vertices, triangles, periods=5):
    """One module carried around the torus by the five-fold rotation."""
    period = 2.0 * math.pi / periods
    all_vertices, all_triangles = [], []
    for k in range(periods):
        angle = k * period
        c, s = math.cos(angle), math.sin(angle)
        rotated = vertices.copy()
        rotated[:, 0] = c * vertices[:, 0] - s * vertices[:, 1]
        rotated[:, 1] = s * vertices[:, 0] + c * vertices[:, 1]
        all_vertices.append(rotated)
        all_triangles.append(triangles + k * len(vertices))
    return np.concatenate(all_vertices), np.concatenate(all_triangles)


def wedge_cut(vertices, triangles, span):
    centroid = vertices[triangles].mean(axis=1)
    phi = np.degrees(np.arctan2(centroid[:, 1], centroid[:, 0]))
    keep = ~((phi > span[0]) & (phi < span[1]))
    return triangles[keep]


def build_mesh(name, vertices, triangles, material, smooth):
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(vertices))
    mesh.vertices.foreach_set("co", vertices.ravel())
    mesh.loops.add(3 * len(triangles))
    mesh.loops.foreach_set("vertex_index", triangles.ravel())
    mesh.polygons.add(len(triangles))
    mesh.polygons.foreach_set(
        "loop_start", np.arange(0, 3 * len(triangles), 3, dtype=np.int64)
    )
    mesh.polygons.foreach_set(
        "loop_total", np.full(len(triangles), 3, dtype=np.int64)
    )
    if smooth:
        mesh.polygons.foreach_set("use_smooth", np.ones(len(triangles), dtype=bool))
    mesh.update()
    mesh.validate()
    obj = bpy.data.objects.new(name, mesh)
    obj.data.materials.append(material)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def surface_material(name, colour, metallic, roughness):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    node = material.node_tree.nodes["Principled BSDF"]
    node.inputs["Base Color"].default_value = (*colour, 1.0)
    node.inputs["Metallic"].default_value = metallic
    node.inputs["Roughness"].default_value = roughness
    return material


def emission_material(name, colour, strength):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*colour, 1.0)
    emission.inputs["Strength"].default_value = strength
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def plasma_material():
    """A shell that glows at grazing incidence and lets the bore show through."""
    material = bpy.data.materials.new("plasma")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    layer = nodes.new("ShaderNodeLayerWeight")
    layer.inputs["Blend"].default_value = 0.34
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 0.30, 0.10, 1.0)
    emission.inputs["Strength"].default_value = 1.8
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(layer.outputs["Facing"], mix.inputs["Fac"])
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(emission.outputs["Emission"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    material.blend_method = "BLEND"
    return material


def add_field_lines(bundle, material):
    for entry in bundle["field_lines"]:
        points = decode_array(entry["points"])
        curve = bpy.data.curves.new(f"line{entry['line']}", type="CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = 0.0075
        curve.bevel_resolution = 3
        spline = curve.splines.new("POLY")
        spline.points.add(len(points) - 1)
        flat = np.concatenate(
            [points, np.ones((len(points), 1))], axis=1
        ).ravel()
        spline.points.foreach_set("co", flat)
        obj = bpy.data.objects.new(f"line{entry['line']}", curve)
        obj.data.materials.append(material)
        bpy.context.scene.collection.objects.link(obj)


def add_strikes(bundle, material):
    points = decode_array(bundle["strikes"]["points"])
    mesh = bpy.data.meshes.new("strikes")
    mesh.from_pydata(points.tolist(), [], [])
    obj = bpy.data.objects.new("strikes", mesh)
    bpy.context.scene.collection.objects.link(obj)
    sphere = bpy.data.meshes.new("spark")
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=1, radius=0.02)
    bm.to_mesh(sphere)
    bm.free()
    spark = bpy.data.objects.new("spark", sphere)
    spark.data.materials.append(material)
    bpy.context.scene.collection.objects.link(spark)
    spark.parent = obj
    obj.instance_type = "VERTS"
    spark.hide_render = False


def add_light(name, location, rotation, size, power, colour):
    light = bpy.data.lights.new(name, type="AREA")
    light.size = size[0]
    light.size_y = size[1]
    light.shape = "RECTANGLE"
    light.energy = power
    light.color = colour
    obj = bpy.data.objects.new(name, light)
    obj.location = location
    obj.rotation_euler = rotation
    bpy.context.scene.collection.objects.link(obj)


def look_at(obj, target):
    direction = np.asarray(target) - np.asarray(obj.location)
    from mathutils import Vector

    obj.rotation_euler = (
        Vector(direction).to_track_quat("-Z", "Y").to_euler()
    )


def main():
    scene = bpy.context.scene
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    manifest = json.loads(MESHES.read_text())["components"]
    for name, (colour, metallic, roughness, smooth, wedge) in SURFACES.items():
        block = manifest.get(name)
        if block is None:
            print(f"{name}: absent from the manifest")
            continue
        vertices, triangles = replicate(*decode_block(block))
        if wedge is not None:
            triangles = wedge_cut(vertices, triangles, wedge)
        material = surface_material(name, colour, metallic, roughness)
        build_mesh(name, vertices, triangles, material, smooth)
        print(f"{name}: {len(triangles)} triangles")

    bundle = json.loads(BUNDLE.read_text())
    plasma_block = manifest.get("plasma")
    if plasma_block is not None:
        vertices, triangles = replicate(*decode_block(plasma_block))
        build_mesh("plasma", vertices, triangles, plasma_material(), True)

    add_field_lines(bundle, emission_material("filament", (1.0, 0.42, 0.12), 4.0))
    add_strikes(bundle, emission_material("spark", (1.0, 0.72, 0.30), 22.0))

    world = bpy.data.worlds.new("void")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (0.004, 0.006, 0.010, 1.0)
    background.inputs["Strength"].default_value = 1.0
    scene.world = world

    add_light("key", (10.5, -7.0, 6.0), (math.radians(55), 0, math.radians(56)),
              (7.0, 3.5), 2600.0, (1.0, 0.96, 0.90))
    add_light("rim", (-9.0, 8.0, 4.5), (math.radians(60), 0, math.radians(-131)),
              (6.0, 3.0), 1400.0, (0.75, 0.85, 1.0))
    add_light("fill", (7.5, 6.5, -3.5), (math.radians(115), 0, math.radians(49)),
              (4.0, 2.0), 420.0, (1.0, 0.85, 0.70))

    camera_data = bpy.data.cameras.new("camera")
    camera_data.lens = 40.0
    camera_data.sensor_width = 36.0
    camera = bpy.data.objects.new("camera", camera_data)
    camera.location = (21.2, -5.5, 9.3)
    scene.collection.objects.link(camera)
    look_at(camera, (-0.9, 0.4, -1.0))
    scene.camera = camera
    camera_data.dof.use_dof = True
    camera_data.dof.focus_distance = 18.5
    camera_data.dof.aperture_fstop = 5.6

    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type in ("OPTIX", "CUDA")
    scene.cycles.samples = 96 if PREVIEW else 768
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1300 if PREVIEW else 2600
    scene.render.resolution_y = 520 if PREVIEW else 1040
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Punchy"
    scene.view_settings.exposure = 0.35
    scene.render.filepath = str(OUT)
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 90

    bpy.ops.render.render(write_still=True)
    print(f"wrote {OUT}")


main()
