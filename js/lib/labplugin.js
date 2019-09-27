var plugin = require('./index');
var base = require('@jupyter-widgets/base');

module.exports = {
  id: 'ipysliceviewer',
  requires: [base.IJupyterWidgetRegistry],
  activate: function(app, widgets) {
      widgets.registerWidget({
          name: 'ipysliceviewer',
          version: plugin.version,
          exports: plugin
      });
  },
  autoStart: true
};

