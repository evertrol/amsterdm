import param
import panel as pn

class ActionExample(param.Parameterized):
    """
    Demonstrates how to use param.Action to trigger an update.
    """

    action = param.Action(default=lambda x: x.param.trigger('action'), label='Click here!')

    number = param.Integer(default=0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #self.param.watch(self.get_number, "action")
    @param.depends('action')
    def get_number(self):
        self.number += 1
        print('triggered')
        return f'Number: {self.number}'

    def view(self):
        return pn.Row(
            pn.Column(
                self.param.action,
                self.param.number,
#                pn.panel(self, show_name=False, margin=0, widgets={"action": {"button_type": "primary"}, "number": {"disabled": True}}),
                '**Click the button** to trigger an update in the output.'
            ),
            self.get_number,
        )


example = ActionExample()

layout = example.view()

pn.serve(
    layout,
    title="Example plot",
    port=5006,
    show=False,
    autoreload=True,
    dev=True,
)
