"""
raycaster.py

# Copyright (c) 2019, Minho Kim
# Computer Graphics Lab, Dept of Computer Science, University of Seoul
# All rights reserved.

"""
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
import pickle
import numpy as np
from math import ceil, log, tan, cos, sin
import os
from glfw import *
#from pyglfw.libapi import *

class VolumeInfo:
    def __init__(self, filename, dtype, dim, scale, level, inverted):
        self.filename = filename
        self.dtype = dtype
        self.dim = dim
        self.scale = scale
        self.level = level
        self.inverted = inverted

###################################################################################################################
class BBox:
#------------------------------------------------------------------------------------------------------------------------ 
    def __init__(self, dim, scale, size_fbo):
        self.dim = np.array(dim)
        self.fbo = FBO_bbox(size_fbo[0], size_fbo[1])
        # - Shaders to render the bounding box containing the whole volume.
        # - Used to set the starting/ending point of each ray.
        # - Less efficient than using `bbox_minmax.*' shaders.
        # - Used when `minmax' parameter of `render' function is FALSE.
        self.prog_bbox = Shader('bbox.vert', 'bbox.frag', ['MVP', 'scale'])  

        # - Actual size of the bounding volume.
        self.size = tuple((self.dim[i])*scale[i] for i in range(3))
        # - The scaling of the bounding box.
        # - We obtain the properly scaled bounding box by applying this to a unit cube.
        self.scale_bbox = tuple(self.size[i]/max(self.size) for i in range(3))
        # - Used to convert from [0,1]^3 space to the lattice space.
        # - Passed as `scale_axes' to raycasting shader.
#        self.scale_axes = tuple(((self.dim[i])*self.size[i])/max(self.size) for i in range(3))
        self.scale_axes = np.array(tuple(((self.dim[i])*self.size[i])/max(self.size) for i in range(3)))
        positions = np.array([  0, 0, 1,
                                1, 0, 1,
                                1, 1, 1,
                                0, 1, 1,
                                0, 0, 0,
                                1, 0, 0,
                                1, 1, 0,
                                0, 1, 0],
                                dtype=np.float32)
        indices = np.array([    0, 1, 2, 2, 3, 0, # front
                                1, 5, 6, 6, 2, 1, # top
                                7, 6, 5, 5, 4, 7, # back
                                4, 0, 3, 3, 7, 4, # bottom
                                4, 5, 1, 1, 0, 4, # left
                                3, 2, 6, 6, 7, 3 # right
                                ], dtype=np.int8)
        # Setting up the VAO for the bbox
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        self.vbo_position = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_position)
        glBufferData(GL_ARRAY_BUFFER, len(positions)*ctypes.sizeof(ctypes.c_float), positions, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

        self.vbo_idx = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.vbo_idx)
        self.size_indices = len(indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(indices)*ctypes.sizeof(ctypes.c_ubyte), indices, GL_STATIC_DRAW)

        glBindVertexArray(0)
#------------------------------------------------------------------------------------------------------------------------ 
    # Render the bounding box to set the starting or ending position of of each ray.
    # Called by render_backfaces() and render_frontfaces()
    def render(self, MVP):
        glUseProgram(self.prog_bbox.id)
        glUniformMatrix4fv(self.prog_bbox.uniform_locs['MVP'], 1, GL_TRUE, MVP)
        glUniform3fv(self.prog_bbox.uniform_locs['scale'], 1, self.scale_bbox)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.size_indices, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        glBindVertexArray(0)
        glUseProgram(0)
#------------------------------------------------------------------------------------------------------------------------ 
    def render_backfaces(self, MVP):
        glDepthFunc(GL_GREATER)
        glClearDepth(0)
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)    # cull the front-facing triangles
        self.render(MVP)
        glDisable(GL_CULL_FACE)
#------------------------------------------------------------------------------------------------------------------------ 
    def render_frontfaces(self, MVP):
        glDepthFunc(GL_LESS)
        glClearDepth(1)
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT )
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK) # cull the back-facing triangles
        self.render(MVP)
        glDisable(GL_CULL_FACE)
#------------------------------------------------------------------------------------------------------------------------ 
    def render_bbox(self, MVP):
        glViewport(0, 0, self.fbo.width, self.fbo.height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.fbo.buf_back, 0)
        self.render_backfaces(MVP)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.fbo.buf_front, 0)
        self.render_frontfaces(MVP)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
###################################################################################################################
class Volume:
    def __init__(self, info):
        self.info = info

        data = np.fromfile(info.filename, dtype=info.dtype).astype(np.float32)

        self.texid = glGenTextures(1)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glBindTexture(GL_TEXTURE_3D, self.texid)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, info.dim[0], info.dim[1], info.dim[2], 0, GL_RED, GL_FLOAT, data)
        glBindTexture(GL_TEXTURE_3D, 0)
        data = None
###################################################################################################################
class UtilMat:
    def perspective(fovy, aspect, n, f):
        t = tan(fovy/2)
        P = np.array([  [1/(aspect*t),  0 ,       0     , 0],
                        [     0      , 1/t,       0     , 0],
                        [     0      ,  0 , -(f+n)/(f-n),-1],
                        [     0      ,  0 , -2*f*n/(f-n), 0]])
        return P
#------------------------------------------------------------------------------------------------------------------------ 
    def ortho(l, r, b, t, n, f):
        P = np.array([  [2/(r-l),    0   ,    0    , -(r+l)/(r-l)], 
                        [   0   , 2/(t-b),    0    , -(t+b)/(t-b)],
                        [   0   ,    0   , -2/(f-n), -(f+n)/(f-n)],
                        [   0   ,    0   ,    0    ,       1     ]])
        return P
#------------------------------------------------------------------------------------------------------------------------ 
    def rotation(angle, axis):
        # https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
        x,y,z = axis
        c,s = cos(angle), sin(angle)
        R = np.array([[ c+x**2*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0],
                      [y*x*(1-c)+z*s,  c+y**2*(1-c), y*z*(1-c)-x*s, 0],
                      [z*x*(1-c)-y*s, z*y*(1-c)+x*s,  c+z**2*(1-c), 0],
                      [            0,             0,             0, 1]])
        return R
#------------------------------------------------------------------------------------------------------------------------ 
    def translation(offset):
        x,y,z = offset
        T = np.array([[1,0,0,x], [0,1,0,y], [0,0,1,z], [0,0,0,1]])
        return T
###################################################################################################################
class FBO_bbox: # FBOs used as render target
    def __init__(self, width, height):
        self.width = width
        self.height = height

        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        self.buf_back = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.buf_back)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.buf_front = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.buf_front)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glBindTexture(GL_TEXTURE_2D, 0)

        self.rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.rbo)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
###################################################################################################################
class FBO_target:
    def __init__(self, size, num_targets):
        self.width, self.height = size
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self.texids = glGenTextures(num_targets)
        self.buffers = [GL_COLOR_ATTACHMENT0+i for i in range(num_targets)]
        for i in range(num_targets):
            glBindTexture(GL_TEXTURE_2D, self.texids[i])
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.width, self.height, 0, GL_RGBA, GL_FLOAT, None)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+i, GL_TEXTURE_2D, self.texids[i], 0)
            glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

###################################################################################################################
class QuadFull:
    def __init__(self, volume, bbox, size_fbo):
        self.init_colormap()
        self.fbo = FBO_target(size_fbo, 4)
        self.bbox = bbox
        # texture to store starting/ending ray positions
        self.tex_bbox_back = bbox.fbo.buf_back
        self.tex_bbox_front = bbox.fbo.buf_front
        self.tex_volume = volume.texid # 3d volume texture
        self.prog_raycast = Shader('raycast_simple.vert', 'cc7_raycast.frag', 
                ['tex_back', 'tex_front', 'tex_volume', 'scale_axes', 'dim', 'level', 'MV'])
        self.progs_shading = []
        uniforms = ['tex_position', 'tex_gradient', 'tex_Hessian1', 'tex_Hessian2', 'tex_colormap', 'tex_colormap_2d', 'MV']
        self.progs_shading.append(('Blinn-Phong', Shader('raycast_simple.vert', 'cc7_shading_BP.frag', uniforms)))
        self.progs_shading.append(('curvature', Shader('raycast_simple.vert', 'cc7_shading_curvature.frag', uniforms)))
        self.init_vao()
        self.idx_shading = 1
        self.scale_step = 0.001
#------------------------------------------------------------------------------------------------------------------------ 
    def init_vao(self):
        # full-screen quad to trigger raycasting
        verts = np.array(
            [-1, -1, 0, 0,
              1, -1, 1, 0,
              1,  1, 1, 1,
             -1,  1, 0, 1], dtype=np.float32)
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, len(verts)*ctypes.sizeof(ctypes.c_float), verts, GL_STATIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*ctypes.sizeof(ctypes.c_float), None)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*ctypes.sizeof(ctypes.c_float), ctypes.c_void_p(2*ctypes.sizeof(ctypes.c_float)))
        glBindVertexArray(0)
#------------------------------------------------------------------------------------------------------------------------ 
    def render_raycast(self, level, volume, MV):
        glViewport(0, 0, self.fbo.width, self.fbo.height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo.fbo)
        glDrawBuffers(len(self.fbo.texids), self.fbo.buffers)
        glClearColor(0, 0, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex_bbox_back)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.tex_bbox_front)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_3D, self.tex_volume)

        prog = self.prog_raycast
        glUseProgram(prog.id)
        glUniform1i(prog.uniform_locs['tex_back'], 0)   
        glUniform1i(prog.uniform_locs['tex_front'], 1) 
        glUniform1i(prog.uniform_locs['tex_volume'], 2)
        glUniform1f(prog.uniform_locs['level'], level)
        glUniform3fv(prog.uniform_locs['scale_axes'], 1, self.bbox.scale_axes)
        glUniform3fv(prog.uniform_locs['dim'], 1, volume.info.dim)
        glUniformMatrix4fv(prog.uniform_locs['MV'], 1, GL_TRUE, MV)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4)
        glBindVertexArray(0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
#------------------------------------------------------------------------------------------------------------------------ 
    def render_deferred_shading(self, MV): # per-pixel deferred shading
        glClearColor(1,1,1,1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#        # bind four textures (0: position 1: gradient 2,3: Hessian)
        for i in range(4):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D, self.fbo.texids[i])
        # bind the 2d color map texture for curvature shading
        glActiveTexture(GL_TEXTURE0 + 4)
        glBindTexture(GL_TEXTURE_2D, self.tex_cm)
        prog = self.progs_shading[self.idx_shading][1]
        glUseProgram(prog.id)
        glUniform1i(prog.uniform_locs['tex_position'], 0)
        glUniform1i(prog.uniform_locs['tex_gradient'], 1)
        glUniform1i(prog.uniform_locs['tex_Hessian1'], 2)
        glUniform1i(prog.uniform_locs['tex_Hessian2'], 3)
        glUniform1i(prog.uniform_locs['tex_colormap_2d'], 4)
        glUniformMatrix4fv(prog.uniform_locs['MV'], 1, GL_TRUE, MV)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4) 
        glBindVertexArray(0)
#------------------------------------------------------------------------------------------------------------------------ 
    def init_colormap(self):
        # colormap for min-max curvature
        cm = np.array([[ 1, 0, 0], [ 1, 1, 0], [0,1,0],
                       [.5,.5,.5], [.5,.5,.5], [0,1,1],
                       [.5,.5,.5], [.5,.5,.5], [0,0,1]], dtype=np.float32)
        self.tex_cm = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_cm)
        glPixelStorei(GL_UNPACK_ALIGNMENT,1)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 3, 3, 0, GL_RGB, GL_FLOAT, cm)
###################################################################################################################
class Shader:
    def __init__(self, filename_vert, filename_frag, uniforms):
        src_vert = self.load_source(filename_vert)
        src_frag = self.load_source(filename_frag)
        self.id = self.build(src_vert, src_frag, uniforms)
#------------------------------------------------------------------------------------------------------------------------ 
    def load_source(self, filename):
        # If the shader source is composed of several files, merge them.
        if isinstance(filename, list):
            src = ''
            for fn in filename:
                src += open(fn, 'r').read()
        else:   # Otherwise, just read the source file.
            src = open(filename, 'r').read()
        return src
#------------------------------------------------------------------------------------------------------------------------ 
    def compile(self, src, type):
        id = glCreateShader(type)
        glShaderSource(id, src)
        glCompileShader(id)
        result = glGetShaderiv(id, GL_COMPILE_STATUS)
        if not(result):
            print('shader compilation error.')
            print(glGetShaderInfoLog(id))
            input('press any key to continue.')
            raise RuntimeError(
                """Shader compile failure (%s): %s"""%(result, glGetShaderInfoLog( id ),),
                src, type,)
        return id
#------------------------------------------------------------------------------------------------------------------------ 
    def build(self, src_vert, src_frag, uniforms):
        id_vert = self.compile(src_vert, GL_VERTEX_SHADER)
        id_frag = self.compile(src_frag, GL_FRAGMENT_SHADER)
        program = glCreateProgram()
        if not program:
            raise RunTimeError('glCreateProgram faled!')
        glAttachShader(program, id_vert)
        glAttachShader(program, id_frag)
        glLinkProgram(program)
        status = glGetProgramiv(program, GL_LINK_STATUS)
        if not status:
            infoLog = glGetProgramInfoLog(program)
            glDeleteProgram(program)
            glDeleteShader(id_vert)
            glDeleteShader(id_frag)
            print(infoLog)
            raise RuntimeError("Error linking program:\n%s\n", infoLog)
        self.uniform_locs = {}
        for u in uniforms:
            self.uniform_locs[u] = glGetUniformLocation(program, u)
        return program
###################################################################################################################
class Scene:    
    # initialization
    def __init__(self, width, height):
        self.width, self.height = width, height

        self.view_angle = 10
        self.angle_x, self.angle_y = 0, 0
        self.position_x, self.position_y = 0, 0

        fbo_size = 512, 512      # framebuffer size

        volinfo = VolumeInfo('./dragon128f.raw', np.float32, (128, 128, 128), (1.2,0.845973,0.53661), 0, False)

        self.bbox = BBox(volinfo.dim, volinfo.scale, fbo_size)
        self.volume = Volume(volinfo)
        self.quad_full = QuadFull(self.volume, self.bbox, fbo_size)
        self.level = volinfo.level

        self.refresh_MVP()
#------------------------------------------------------------------------------------------------------------------------ 
    def refresh_MVP(self):
        Rx = UtilMat.rotation(np.radians(self.angle_x), [1,0,0])
        Ry = UtilMat.rotation(np.radians(self.angle_y), [0,1,0])
        self.P = UtilMat.perspective(np.radians(self.view_angle), self.width/self.height, 1, 3)
        T = UtilMat.translation([self.position_x,self.position_y,-2])
        self.MV = np.dot(T,np.dot(Rx, Ry))
        self.MVP = np.dot(self.P,self.MV)
#------------------------------------------------------------------------------------------------------------------------ 
    def render(self):
        # step 1: render the bounding box onto the framebuffers to compute the entering/exiting positions.
        self.bbox.render_bbox(self.MVP)
        # step 2: start raycasting and render onto the framebuffer
        self.quad_full.render_raycast(self.level, self.volume, self.MV) 
        # Since the viewport has been changed during raycasting, restore the viewport.
        glViewport(0, 0, self.width, self.height)   
        # step 3: copy the content of the framebuffer to the window
        self.quad_full.render_deferred_shading(self.MV)
###################################################################################################################
class RenderWindow:
    def __init__(self):
        cwd = os.getcwd() # save current working directory
        glfwInit() # initialize glfw - this changes cwd
        os.chdir(cwd) # restore cwd

        # version hints
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)
        glfwWindowHint(GLFW_OPENGL_PROFILE, 
                            GLFW_OPENGL_CORE_PROFILE)
        self.width, self.height = 512, 512
        self.aspect = self.width/float(self.height)
        self.win = glfwCreateWindow(self.width, self.height, b'raycaster')
        # make context current
        glfwMakeContextCurrent(self.win)

        print("OpenGL version = ", glGetString( GL_VERSION ))
        print("GLSL version = ", glGetString( GL_SHADING_LANGUAGE_VERSION ))
        
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.0, 0.0, 0.0,0.0)
        glfwSetKeyCallback(self.win, self.onKeyboard)
        glfwSetWindowSizeCallback(self.win, self.onSize)        
        self.scene = Scene(self.width, self.height)
        self.exitNow = False
#------------------------------------------------------------------------------------------------------------------------ 
    def onKeyboard(self, win, key, scancode, action, mods):
        if action == GLFW_PRESS:
            if key == GLFW_KEY_ESCAPE: # ESC to quit
                self.exitNow = True
            elif key == GLFW_KEY_RIGHT:    # right arrow
                if mods & GLFW_MOD_SHIFT:
                    self.scene.position_x += .1;
                else:
                    self.scene.angle_y = (self.scene.angle_y + 10) % 360
                self.scene.refresh_MVP()
            elif key == GLFW_KEY_LEFT:
                if mods & GLFW_MOD_SHIFT:
                    self.scene.position_x -= .1;
                else:
                    self.scene.angle_y = (self.scene.angle_y - 10) % 360
                self.scene.refresh_MVP()
            elif key == GLFW_KEY_UP:
                if mods & GLFW_MOD_SHIFT:
                    self.scene.position_y += .1;
                else:
                    self.scene.angle_x = (self.scene.angle_x - 10) % 360
                self.scene.refresh_MVP()
            elif key == GLFW_KEY_DOWN:
                if mods & GLFW_MOD_SHIFT:
                    self.scene.position_y -= .1;
                else:
                    self.scene.angle_x = (self.scene.angle_x + 10) % 360
                self.scene.refresh_MVP()
            elif key == GLFW_KEY_EQUAL:
                self.scene.level = self.scene.level + 0.001
                print(self.scene.level)
            elif key == GLFW_KEY_MINUS:
                self.scene.level = self.scene.level - 0.001
                print(self.scene.level)
            elif key == GLFW_KEY_TAB:
                quad = self.scene.quad_full
                quad.idx_shading = (quad.idx_shading + 1)%len(quad.progs_shading)
                print('currently rendering: ' + quad.progs_shading[quad.idx_shading][0])
            elif key == GLFW_KEY_PAGE_UP:
                self.scene.view_angle = self.scene.view_angle - 1
                self.scene.refresh_MVP()
            elif key == GLFW_KEY_PAGE_DOWN:
                self.scene.view_angle = self.scene.view_angle + 1
                self.scene.refresh_MVP()
#------------------------------------------------------------------------------------------------------------------------ 
    def onSize(self, win, width, height):
        self.aspect = width/float(height)
        self.scene.width = width
        self.scene.height = height
#------------------------------------------------------------------------------------------------------------------------ 
    def run(self):
        glfwSetTime(0)
        glClearColor(1,1,1,1)
        lastT = glfwGetTime()
        frames = 0
        while not glfwWindowShouldClose(self.win) and not self.exitNow:
            currT = glfwGetTime()
            if frames == 20:
                elapsed = currT - lastT
                print('fps = {}'.format(frames/elapsed))
                lastT = currT
                frames = 0
            self.scene.render()
            frames += 1
            glfwSwapBuffers(self.win)
            glfwPollEvents()
        glfwTerminate()
#------------------------------------------------------------------------------------------------------------------------ 
def main():
    print("Starting raycaster. "
          "Press ESC to quit.")
    rw = RenderWindow()
    rw.run()
#------------------------------------------------------------------------------------------------------------------------ 
if __name__ == '__main__':
    main()
