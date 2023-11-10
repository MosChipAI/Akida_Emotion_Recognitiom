import json

from kivy.logger import Logger
import akida


class AkidaSettings():
    """
    Class that handle Akida settings.

    Used to set up settings in configuration panel for the main window.
    """

    def __init__(self, **kwargs):
        # Check available devices
        if len(akida.devices()) > 0:
            self._disable_hw = False
        else:
            self._disable_hw = True

    def set_defaults(self, config, default_model):
        """
        Create configuration default values

        :args:
            config(kivy.Config): User app config object instance
            default_model(str): URL to default model file
        """
        config.setdefaults('akida', {'model': default_model})

        config.setdefaults('device', {'enable_hw': 0})
        config.setdefaults('device', {'power_measure': 0})

    def add_defaults(self, json_data=None, disable_hw_device=False):
        """
        Create akida settings configuration json. By default, method adds
        akida model parameter.
        A boolean parameter could be set to add akida enable/disable HW device
        parameter.

        :args:
            json_data(str): Formatted json string with user app settings.
            disable_hw_device(bool): Disable HW device toggle if user model
            isn't compatible with HW.
            (Default: False)

        :return:
            A formatted json string.
        """

        if json_data == None:
            json_data = """[]"""

        json_data = self.add_model_param(json_data)

        # Manage HW device cases to update toggle on UI
        #  * Device found but model isn't compatible - NOT OK
        #  * Device found and model is compatible - OK
        #  * Device not found and model is compatible/not compatible - NOT OK
        disable_device_option = self._disable_hw or disable_hw_device

        akida_json = """[
        {
            "type": "bool",
            "disabled": """ + str(int(disable_device_option)) + """,
            "title": "HW Device",
            "desc": "Enable/Disable the Akida HW Device",
            "section": "device",
            "key": "enable_hw"
        },
        {
            "type": "bool",
            "disabled": """ + str(int(disable_device_option)) + """,
            "title": "HW Power measures",
            "desc": "Enable/Disable the Akida HW power measures",
            "section": "device",
            "key": "power_measure"
        }]"""

        base_json = json.loads(akida_json)

        for param in json_data:
            base_json.append(param)

        json_data = base_json

        return json.dumps(json_data)

    def add_model_param(self, json_data):
        """
        Add akida model parameter to app json.

        :args:
            json_data(str): Formated json string with user app settings.

        :return:
            A json object with akida model parameter and app settings.
        """
        akida_json = """[
        {
            "type": "string",
            "title": "Akida model",
            "desc": "The Akida model file to use (.h5, .fbz)",
            "section": "akida",
            "key": "model"
        }]"""

        base_json = json.loads(akida_json)
        app_json = json.loads(json_data)

        for param in app_json:
            base_json.append(param)

        return base_json
