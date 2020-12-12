
import gi 
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib


#FIXME: sometimes the mere existence of this class will break a multi-env micropolis run
class ParamRewWindow(Gtk.Window):
    def __init__(self, env, metrics, metric_trgs, metric_bounds):
        self.env = env
        import gi 
        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk, GLib
        Gtk.Window.__init__(self, title="Metrics")
        self.set_border_width(10)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        
        reset_button = Gtk.Button("reset")
        reset_button.connect('clicked', lambda item: self.env.reset())
        hbox.pack_start(reset_button, False, False, 0) 

        auto_reset_button = Gtk.CheckButton("auto reset")
        auto_reset_button.connect('clicked', lambda item: self.env.enable_auto_reset(item))
        auto_reset_button.set_active(True)
        hbox.pack_start(auto_reset_button, False, False, 0)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox.pack_start(hbox, False, False, 0)
        self.add(vbox)

        prog_bars = {}
        scales = {}
        prog_labels = {}
        for k in metrics:
            if k not in self.env.usable_metrics:
                continue
            metric = metrics[k]
            label = Gtk.Label()
            label.set_text(k)
            vbox.pack_start(label, True, True, 0)
            if metric is None:
                metric = 0
            ad = Gtk.Adjustment(metric, metric_bounds[k][0], metric_bounds[k][1],
                                env.param_ranges[k] / 20, env.param_ranges[k] / 10, 0)
            scale = Gtk.HScale(adjustment=ad)
            scale.set_name(k)
            scale.set_show_fill_level(True)
            scales[k] = scale
            vbox.pack_start(scale, True, True, 0)
            scale.connect("value-changed", self.scale_moved)

            prog_label = Gtk.Label()
            prog_label.set_text(str(metric))
            prog_labels[k] = prog_label
            vbox.pack_start(prog_label, True, True, 0)
            metric_prog = Gtk.ProgressBar()
#           metric_prog.set_draw_value(True)
            prog_bars[k] = metric_prog
            vbox.pack_start(metric_prog, True, True, 10)
           #bounds = metric_bounds[k]
           #frac = metrics[k]
           #metric_prog.set_fraction(frac)

      
       #self.timeout_id = GLib.timeout_add(50, self.on_timeout, None)
       #self.activity_mode = False
        self.prog_bars = prog_bars
        self.scales = scales
        self.prog_labels = prog_labels



    def step(self):
        self.display_metrics()
        while Gtk.events_pending():
            Gtk.main_iteration()

    def scale_moved(self, event):
        k = event.get_name()
        self.env.metric_trgs[k] = event.get_value()
        self.env.set_trgs(self.env.metric_trgs)

    def display_metric_trgs(self):
        for k, v in self.env.metric_trgs.items():
            if k in self.env.usable_metrics:
                self.scales[k].set_value(v)

    def display_metrics(self):
        for k, prog_bar in self.prog_bars.items():
            metric_val = self.env.metrics[k]
            prog_bar.set_fraction(metric_val / self.env.param_ranges[k])
            prog_label = self.prog_labels[k]
            prog_label.set_text(str(metric_val))

    def on_show_text_toggled(self, button):
        show_text = button.get_active()
        if show_text:
            text = "some text"
        else:
            text = None
        self.progressbar.set_text(text)
        self.progressbar.set_show_text(show_text)

    def on_activity_mode_toggled(self, button):
        self.activity_mode = button.get_active()
        if self.activity_mode:
            self.progressbar.pulse()
        else:
            self.progressbar.set_fraction(0.0)

    def on_right_to_left_toggled(self, button):
        value = button.get_active()
        self.progressbar.set_inverted(value)

    def on_timeout(self, user_data):
        """
        Update value on the progress bar
        """
        if self.activity_mode:
            self.progressbar.pulse()
        else:
            new_value = self.progressbar.get_fraction() + 0.01

            if new_value > 1:
                new_value = 0

            self.progressbar.set_fraction(new_value)

        # As this is a timeout function, return True so that it
        # continues to get called
        return True

