import numpy
import threading
import time
import roslib
try:
	roslib.load_manifest('TFG-Bebop-YOLO')
except:
	pass
import rospy
from std_msgs.msg import Empty
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from bebop_msgs.msg import CommonCommonStateBatteryStateChanged as Battery
import termios
import sys, tty


#-- Lectura inmediata por teclado
def getch():
    def _getch():
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
    return _getch()

class ReactiveBebop:
	
	def print_help(self, h_type):
		if h_type==0: # Ayuda para menu
			print('\n\n\n\
[0] Modo manual \n\
[1] Seleccionar target video principal (solo funciona en modo automatico)\n\
[2] Seleccionar target video secundario \n\
[3] Seleccionar velocidad [0.0 - 1.0]\n\
[4] Empezar/Parar grabacion\n\
[5] Mostrar % bateria\n\
[p] Parada de emergencia\n\
[q] Salir del programa\n\
[h] Help\n\
\n\n\n')

		elif h_type==1: # Ayuda para movimiento
			print('\n\n\n\
[ ] Despega\n\
[l] Aterrizar\n\
[w] Avanza\n\
[a] Desplaza izda\n\
[s] Retrocede\n\
[d] Desplaza dcha\n\
[8] Asciende\n\
[4] Gira izda\n\
[6] Gira dcha\n\
[2] Desciende\n\
[5] Para\n\
[e] Exit. Modo auto\n\
[p] Parada de emergencia\n\
[h] Help\n\
\n\n\n')

	def __init__(self, res, target=None, tracks=[], current_track=None):
		self.target = target #id_track
		self.tracks = tracks
		self.current_track = current_track
		self.cam_tracks = []
		self.current_cam_track = None
		self.min_x = 0.4*res[0]
		self.max_x = 0.6*res[0]
		self.min_y = 0.4*res[1]
		self.max_y = 0.6*res[1]
		self.auto = True
		self.t_menu = True
		self.vel_av = 0.1
		self.vel_giro = 0.07
		self.min_height = 60.0
		self.max_height = 80.0
		self.record = False
		self.show_battery = False
		self.battery = 'unknown'
		self.aux=0
		
		#-- Topics
		self.takeoff_pub = rospy.Publisher('/bebop/takeoff',Empty, queue_size=10)
		self.land_pub = rospy.Publisher('/bebop/land', Empty, queue_size=10)
		self.reset_pub = rospy.Publisher('/bebop/reset', Empty, queue_size=10)
		self.move_pub = rospy.Publisher('/bebop/cmd_vel', Twist, queue_size=10)
		self.video_pub = rospy.Publisher('/bebop/record', Bool, queue_size=10)
		self.battery_sub = rospy.Subscriber('/bebop/states/common/CommonState/BatteryStateChanged',Battery,self.battery_callback)

	def battery_callback(self, data):
		if not self.show_battery:
			self.battery = str(data).split(' ')[-1]

	def update_tracks(self, confirmed_tracks):
		self.tracks=confirmed_tracks

	def update_cam_tracks(self, confirmed_tracks):
		self.cam_tracks=confirmed_tracks

		if self.current_cam_track<>None:
			if self.current_cam_track.state<>2:
				self.move([' '])
				self.current_cam_track=None
	
	def update_target(self, target):
		self.target=target
		
	def move(self, moves=[]):
		msg=Twist()
		msg.linear.z = 0
		msg.angular.z = 0
		msg.linear.x = 0
		msg.linear.y = 0

		for move in moves:
			if move==' ': # despegar
				self.takeoff_pub.publish(Empty())
				#print('move: '+move)
			elif move=='l': # aterrizar
				self.land_pub.publish(Empty())
				#print('move: '+move)
			elif move=='w': # avanzar
				msg.linear.x = self.vel_av
				#print('move: '+move)
			elif move=='a': # desplazar a la izda
				msg.linear.y = self.vel_av	
				#print('move: '+move)
			elif move=='s': # retroceder
				msg.linear.x = -self.vel_av
				#print('move: '+move)
			elif move=='d': # desplazar a la derecha
				msg.linear.y = -self.vel_av
				#print('move: '+move)
			elif move=='8': # ascender
				msg.linear.z = self.vel_giro
				#print('move: '+move)
			elif move=='4': # rotar izda
				msg.angular.z = self.vel_giro
				#print('move: '+move)
			elif move=='6': # rotar dcha
				msg.angular.z = -self.vel_giro
				#print('move: '+move)
			elif move=='2': # descender
				msg.linear.z = -self.vel_giro
				#print('move: '+move)
			elif move=='5': # para
				msg.linear.z = 0
				#print('move: '+move)
				msg.angular.z = 0
				msg.linear.x = 0
				msg.linear.y = 0
			elif move=='e': # cambiar de modo
				self.auto=True
				#print('move: '+move)
				self.print_help(0)
			elif move=='p': # parada de emergencia
				self.reset_pub.publish(Empty())
				#print('move: '+move)
			elif move=='h': # ayuda
				self.print_help(1)
				#print('move: '+move)
		
		# Si no mandamos un mensaje cada 0.1s
		# el dron detecta que hay un error
		# y se mantiene estatico
		if moves:
			self.move_pub.publish(msg)
	
	def follow_target(self):
		#time.sleep(3) # Damos tiempo a que carguen el resto de hebras
		it=0
		while(self.auto):
			moves=[]
			if self.current_track != None:
				if self.current_track.state == 2:
					minx, miny, maxx, maxy = self.current_track.to_tlbr()
					centroid = (int(minx+maxx/2),int(miny+maxy/2))
					h = maxy-miny
					w = maxx-minx
					
					if centroid[0] < self.min_x:
						moves.append('4')
					elif centroid[0] > self.max_x:
						moves.append('6')
					"""
					if centroid[1] < self.min_y:
						moves.append('8')
					elif centroid[1] > self.max_y:
						moves.append('2')	
					"""
					if h < self.min_height:
						moves.append('w')
					elif h > self.max_height:
						moves.append('s')

					if not moves:
						moves.append('5')

				else: #Si hemos perdido el track
					if it==1:
						it=-1
						moves.append('w')
					else:
						moves.append('4')

					it+=1
				
			self.move(moves)
		
	def menu(self, option):
		print(option)	
		if option=='0': # modo manual
			self.auto = False
			try:
				if self.follow_thread.is_alive():
					self.follow_thread.join()
			except: pass
			c=''
			self.print_help(1) # Imprime ayuda movimiento
			
			while(c!='e'):
				c=getch()
				try:
					self.move([c]) # Realiza movimiento 
				except:
					print('Tecla no valida')
					self.move('h')
				
		elif option=='1': # Seleccion de target
			ids = [t.track_id for t in self.tracks]
			print('\'-1\'Exit.\n Select target: ')
			try:
				new_target = input()
			except:
				new_target = -2
			
			while( (not new_target in ids) and new_target != -1):
				print('Bad Target. Select one of this: ')
				ids = [t.track_id for t in self.tracks]
				print(ids)
				print('\'-1\'Exit.\n Select target: ')
				try:
					new_target = input()
				except:
					new_target = -2
			
			if new_target != -1:
				self.auto = True
				try:
					self.follow_thread = threading.Thread(target=self.follow_target, args=())
					self.follow_thread.daemon=True
					self.follow_thread.start()
				except:
					pass
				self.target = new_target
				self.current_track = [tr for tr in self.tracks if tr.track_id==new_target][0]
				print('Target Updated')

		elif option=='2': # Seleccion de target
			if self.cam_tracks!=[]:
				ids = [t.track_id for t in self.cam_tracks]
				print('\'-1\'Exit.\n Select target: ')
				try:
					new_target = input()
				except:
					new_target = -2
				
				while( (not new_target in ids) and new_target != -1):
					print('Bad Target. Select one of this: ')
					ids = [t.track_id for t in self.cam_tracks]
					print(ids)
					print('\'-1\'Exit.\n Select target: ')
					try:
						new_target = input()
					except:
						new_target = -2
				
				if new_target != -1:
					self.current_cam_track = [tr for tr in self.cam_tracks if tr.track_id==new_target][0]
					print('Target Updated')
			else:
				print('No hay video secundario o bien no se han detectado tracks en el\n')
			
		elif option=='3': # Seleccion de velocidad
			print('Velocidad actual de giro: '+str(self.vel_giro)+'\nIndique nueva velocidad [0.0 - 1.0]: ')
			try:
				v = input()
				if v>=0 and v<=1:
					self.vel_giro = v
					print('Velocidad actualizada')
				else:
					print('Velocidad fuera de los limites')
			except:
				print('Error en la entrada de velocidad')
			print('Velocidad lineal actual: '+str(self.vel_av)+'\nIndique nueva velocidad [0.0 - 1.0]: ')
			try:
				v = input()
				if v>=0 and v<=1:
					self.vel_av = v
					print('Velocidad actualizada')
				else:
					print('Velocidad fuera de los limites')
			except:
				print('Error en la entrada de velocidad')
			
		elif option=='4': # Empezar/Parar grabacion
			if not self.record:
				self.record = True
				self.video_pub.publish(True)
				print('Ha comenzado la grabacion\n')
			else:
				self.record = False
				self.video_pub.publish(False)
				print('Se ha detenido la grabacion\n')
			
		elif option=='5': # Mostrar % bateria
			self.show_battery = True
			print('Bateria: '+self.battery+'%')
			self.show_battery = False
			
		elif option=='p': # Parada de emergencia
			self.move([option])
			if self.follow_thread.is_alive():
				self.follow_thread.join(0.0)
			
		elif option=='h': # Parada de emergencia
			self.print_help(0)

	def show_menu(self):
		self.print_help(0)

	def select_menu(self):
		while self.menu:
			option = getch()
			if (option=='0' or option=='1' or option=='2'
				or option=='3' or option=='4' or option=='5' or option=='p'
				or option=='h'):
				self.menu(option)

			if option=='q': break

	def menu_alive(self):
		return self.menu_thread.is_alive()

	def start(self):
		self.menu_thread = threading.Thread(target=self.select_menu, args=())
		self.menu_thread.daemon=True
		self.menu_thread.start()

	def stop(self):
		self.auto=False
		try:
			if self.follow_thread.is_alive():
				self.follow_thread.join()
		except:
			pass
		self.t_menu=False
		if self.menu_thread.is_alive():
			self.menu_thread.join()
















