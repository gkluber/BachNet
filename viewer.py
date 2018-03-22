import os
os.environ['KIVY_GL_BACKEND'] = 'sdl2'

import kivy

kivy.require('1.0.10')
kivy.kivy_configure()

from kivy.app import App
from kivy.uix.button import Button
from kivy.metrics import *

class MIDIViewer(App):
	def on_start(self):
		#todo
		pass
	
	def build(self):
		config = self.config
		#use config.get('section', 'key'), config.getint('section', 'key'), etc. to get the values
		return Button(text="hello uwu")
	
	def build_config(self, config):
		config.setdefaults('section1', {
			'k1':'v1',
			'k2':2
			})
	
	def build_settings(self, settings):
		settings.add_json_panel('MIDI Viewer', self.config, data='''[
            { "type": "title", "title": "MIDI Viewer" }
            ]''')
		settings.interface.menu.width = dp(100)
	
	def on_config_change(self, config, section, key, value):
        if config is self.config:
			pass #TODO
			
	def on_pause(self):
		#todo
		return true
	
	def on_resume(self):
		#todo
		pass
	
	def on_stop(self):
		#todo
		pass
	

class MyKeyboardListener(Widget):

    def __init__(self, **kwargs):
        super(MyKeyboardListener, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(
            self._keyboard_closed, self, 'text')
        if self._keyboard.widget:
            # If it exists, this widget is a VKeyboard object which you can use
            # to change the keyboard layout.
            pass
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

    def _keyboard_closed(self):
        print('My keyboard have been closed!')
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        print('The key', keycode, 'have been pressed')
        print(' - text is %r' % text)
        print(' - modifiers are %r' % modifiers)

        # Keycode is composed of an integer + a string
        # If we hit escape, release the keyboard
        if keycode[1] == 'escape':
            keyboard.release()

        # Return True to accept the key. Otherwise, it will be used by
        # the system.
        return True

	
if __name__ == '__main__':
	MIDIViewer().run()