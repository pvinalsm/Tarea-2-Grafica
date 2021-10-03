# coding=utf-8
"""Tarea 2"""

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.performance_monitor as pm
from grafica.assets_path import getAssetPath

__author__ = "Patricio A. Viñals M."
__license__ = "MIT"

# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True
        self.viewPos = np.array([10,10,10])
        self.camUp = np.array([0, 1, 0])
        self.distance = 10


controller = Controller()

def setPlot(pipeline, mvpPipeline):
    projection = tr.perspective(45, float(width)/float(height), 0.1, 100)

    glUseProgram(mvpPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "lightPosition"), 5, 5, 5)
    
    glUniform1ui(glGetUniformLocation(pipeline.shaderProgram, "shininess"), 1000)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"), 0.001)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"), 0.01)

def setView(pipeline, mvpPipeline):
    view = tr.lookAt(
            controller.viewPos,
            np.array([0,0,0]),
            controller.camUp
        )

    glUseProgram(mvpPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "viewPosition"), controller.viewPos[0], controller.viewPos[1], controller.viewPos[2])
    

def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_LEFT_CONTROL:
        controller.showAxis = not controller.showAxis

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    
    elif key == glfw.KEY_1:
        controller.viewPos = np.array([controller.distance,controller.distance,controller.distance]) #Vista diagonal 1
        controller.camUp = np.array([0,1,0])
    
    elif key == glfw.KEY_2:
        controller.viewPos = np.array([0,0,controller.distance]) #Vista frontal
        controller.camUp = np.array([0,1,0])

    elif key == glfw.KEY_3:
        controller.viewPos = np.array([controller.distance,0,controller.distance]) #Vista lateral
        controller.camUp = np.array([0,1,0])

    elif key == glfw.KEY_4:
        controller.viewPos = np.array([0,controller.distance,0]) #Vista superior
        controller.camUp = np.array([1,0,0])
    
    elif key == glfw.KEY_5:
        controller.viewPos = np.array([controller.distance,controller.distance,-controller.distance]) #Vista diagonal 2
        controller.camUp = np.array([0,1,0])
    
    elif key == glfw.KEY_6:
        controller.viewPos = np.array([-controller.distance,controller.distance,-controller.distance]) #Vista diagonal 2
        controller.camUp = np.array([0,1,0])
    
    elif key == glfw.KEY_7:
        controller.viewPos = np.array([-controller.distance,controller.distance,controller.distance]) #Vista diagonal 2
        controller.camUp = np.array([0,1,0])
    
    else:
        print('Unknown key')

def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

#NOTA: Aqui creas tu escena. En escencia, sólo tendrías que modificar esta función.
def createAvion(pipeline):
    cuerpoNode = sg.SceneGraphNode('cuerpo')
    principal_cuerpo_gpu = createGPUShape(pipeline, bs.createColorCylinderTarea2(0, 0.6, 0))
    cuerpoNode.transform = tr.matmul([tr.translate(-0.7, 0, 0),
                                    tr.scale(3.3, 0.7, 0.7),
                                    tr.rotationZ(np.pi/2)
                                    ])
    cuerpoNode.childs += [principal_cuerpo_gpu]

    cabinaNode = sg.SceneGraphNode('cabina')
    principal_cabina_gpu = createGPUShape(pipeline, bs.createColorCylinderTarea2(1, 1, 1))
    cabinaNode.transform = tr.matmul([tr.translate(0.15, 0.83, 0), 
                            tr.rotationZ(1.9),
                            tr.scale(0.3, 0.01, 0.3)
                            ])
    cabinaNode.childs += [principal_cabina_gpu]

    principalNode = sg.SceneGraphNode('principal')
    principalNode.transform = tr.identity()
    principalNode.childs += [cuerpoNode,
                            cabinaNode]



    alaSupNode = sg.SceneGraphNode('alaSup')
    ala_sup_gpu = createGPUShape(pipeline, bs.createColorCubeTarea2(0, 0.6, 0))
    alaSupNode.transform = tr.matmul([tr.translate(0, 1.6, 0),
                                    tr.scale(0.8, 0.1, 3.8)])
    alaSupNode.childs += [ala_sup_gpu]

    alaInfNode = sg.SceneGraphNode('alaInf')
    ala_inf_gpu = createGPUShape(pipeline, bs.createColorCubeTarea2(0, 0.6, 0))
    alaInfNode.transform = tr.matmul([tr.translate(0, -0.68, 0),
                                    tr.scale(0.8, 0.1, 3.8)])
    alaInfNode.childs += [ala_inf_gpu]

    uniones = createGPUShape(pipeline, bs.createColorCylinderTarea2(0.8, 0.4, 0.0))

    uniones1Node = sg.SceneGraphNode('uniones1')
    uniones1Node.transform = tr.matmul([tr.translate(0.5, 0.44, 2.8),                              
                                        tr.scale(0.03, 1.2, 0.03)])
    uniones1Node.childs += [uniones]

    uniones2Node = sg.SceneGraphNode('uniones2')
    uniones2Node.transform = tr.matmul([tr.translate(-0.5, 0.44, 2.8),                              
                                        tr.scale(0.03, 1.2, 0.03)])
    uniones2Node.childs += [uniones]

    uniones3Node = sg.SceneGraphNode('uniones3')
    uniones3Node.transform = tr.matmul([tr.translate(0.5, 0.44, -2.8),                              
                                        tr.scale(0.03, 1.2, 0.03)])
    uniones3Node.childs += [uniones]

    uniones4Node = sg.SceneGraphNode('uniones4')
    uniones4Node.transform = tr.matmul([tr.translate(-0.5, 0.44, -2.8),                              
                                        tr.scale(0.03, 1.2, 0.03)])
    uniones4Node.childs += [uniones]

    uniones5Node = sg.SceneGraphNode('uniones5')
    uniones5Node.transform = tr.matmul([tr.translate(0.5, 0.5, -0.8),
                                        tr.rotationX(3*np.pi/4),                         
                                        tr.scale(0.03, 1.5, 0.03)])
    uniones5Node.childs += [uniones]

    uniones6Node = sg.SceneGraphNode('uniones6')
    uniones6Node.transform = tr.matmul([tr.translate(-0.5, 0.5, -0.8),
                                        tr.rotationX(3*np.pi/4),                         
                                        tr.scale(0.03, 1.5, 0.03)])
    uniones6Node.childs += [uniones]

    uniones7Node = sg.SceneGraphNode('uniones7')
    uniones7Node.transform = tr.matmul([tr.translate(0.5, 0.5, 0.8),
                                        tr.rotationX(np.pi/4),                         
                                        tr.scale(0.03, 1.5, 0.03)])
    uniones7Node.childs += [uniones]

    uniones8Node = sg.SceneGraphNode('uniones8')
    uniones8Node.transform = tr.matmul([tr.translate(-0.5, 0.5, 0.8),
                                        tr.rotationX(np.pi/4),                         
                                        tr.scale(0.03, 1.5, 0.03)])
    uniones8Node.childs += [uniones]
    
    unionesNode = sg.SceneGraphNode('uniones')
    unionesNode.transform = tr.identity()
    unionesNode.childs += [uniones1Node,
                            uniones2Node,
                            uniones3Node,
                            uniones4Node,
                            uniones5Node,
                            uniones6Node,
                            uniones7Node,
                            uniones8Node]

    alasNode = sg.SceneGraphNode('alas')
    alasNode.transform = tr.matmul([tr.translate(0.8, 0, 0)])
    alasNode.childs += [alaSupNode,
                        alaInfNode,
                        unionesNode]



    blackCylinder = createGPUShape(pipeline, bs.createColorCylinderTarea2(0, 0, 0))

    wheel1Node = sg.SceneGraphNode('wheel1')
    wheel1Node.transform = tr.matmul([tr.translate(0, 0, 1),
                                        tr.rotationX(np.pi/2),                         
                                        tr.scale(0.55, 0.1, 0.55)])
    wheel1Node.childs += [blackCylinder]

    wheel2Node = sg.SceneGraphNode('wheel2')
    wheel2Node.transform = tr.matmul([tr.translate(0, 0, -1),
                                        tr.rotationX(np.pi/2),                         
                                        tr.scale(0.55, 0.1, 0.55)])
    wheel2Node.childs += [blackCylinder]

    unionNode = sg.SceneGraphNode('union')
    ruedas_wheels_union_gpu = createGPUShape(pipeline, bs.createColorCylinderTarea2(1, 1, 1))
    unionNode.transform = tr.matmul([tr.translate(0, 0, 0),
                                        tr.rotationX(np.pi/2),                         
                                        tr.scale(0.1, 0.8, 0.1)])
    unionNode.childs += [ruedas_wheels_union_gpu]

    wheelsNode = sg.SceneGraphNode('wheels')
    wheelsNode.transform = tr.identity()
    wheelsNode.childs += [wheel1Node,
                        wheel2Node,
                        unionNode]

    orangeTriangle = createGPUShape(pipeline, bs.createColorConeTarea2(0.8, 0.4, 0))

    patas1Node = sg.SceneGraphNode('patas1')
    patas1Node.transform = tr.matmul([tr.translate(0, 0, -0.9),
                                        tr.rotationX(np.pi),                         
                                        tr.scale(0.45, 0.45, 0.01)])
    patas1Node.childs += [orangeTriangle]

    patas2Node = sg.SceneGraphNode('patas1')
    patas2Node.transform = tr.matmul([tr.translate(0, 0, 0.9),
                                        tr.rotationX(np.pi),                         
                                        tr.scale(0.45, 0.45, 0.01)])
    patas2Node.childs += [orangeTriangle]

    patasNode = sg.SceneGraphNode('patas')
    patasNode.transform = tr.matmul([tr.translate(0, 0.5, 0)])
    patasNode.childs += [patas1Node,
                            patas2Node]

    ruedasNode = sg.SceneGraphNode('ruedas')
    ruedasNode.transform = tr.matmul([tr.translate(1, -1.74, 0)])
    ruedasNode.childs += [wheelsNode,
                            patasNode]



    tapaNode = sg.SceneGraphNode('tapa')
    llantaNode = sg.SceneGraphNode('llanta')
    tuboNode = sg.SceneGraphNode('tubo')
    helixNode = sg.SceneGraphNode('helix')

    heliceNode = sg.SceneGraphNode('helice')
    heliceNode.transform = tr.identity()
    heliceNode.childs += [
        tapaNode,
        llantaNode,
        tuboNode,
        helixNode
    ]

    timonSupNode = sg.SceneGraphNode('timonSup')
    timonInfNode = sg.SceneGraphNode('timonInf')
    timonDerNode = sg.SceneGraphNode('timonDer')
    timonIzqNode = sg.SceneGraphNode('timonIzq')

    timonesNode = sg.SceneGraphNode('timones')
    timonesNode.transform = tr.identity()
    timonesNode.childs += [
        timonSupNode,
        timonInfNode,
        timonDerNode,
        timonIzqNode
    ]

    avion = sg.SceneGraphNode('system')
    avion.transform = tr.identity()
    avion.childs += [principalNode,
        alasNode,
        ruedasNode,
        heliceNode,
        timonesNode
    ]
    
    return avion

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "Tarea 2"
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    pipeline = ls.SimpleGouraudShaderProgram()
    
    # Telling OpenGL to use our shader program
    glUseProgram(mvpPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    cpuAxis = bs.createAxis(7)
    gpuAxis = es.GPUShape().initBuffers()
    mvpPipeline.setupVAO(gpuAxis)
    gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)

    #NOTA: Aqui creas un objeto con tu escena
    dibujo = createAvion(pipeline)

    setPlot(pipeline, mvpPipeline)

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)

    while not glfw.window_should_close(window):

        # Measuring performance
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))

        # Using GLFW to check for input events
        glfw.poll_events()

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        setView(pipeline, mvpPipeline)

        if controller.showAxis:
            glUseProgram(mvpPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            mvpPipeline.drawCall(gpuAxis, GL_LINES)

        #NOTA: Aquí dibujas tu objeto de escena
        glUseProgram(pipeline.shaderProgram)
        sg.drawSceneGraphNode(dibujo, pipeline, "model")
        

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # freeing GPU memory
    gpuAxis.clear()
    dibujo.clear()
    

    glfw.terminate()