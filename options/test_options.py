from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--encoder_path', type=str, default='', help='path for trained encoder')
        self.parser.add_argument('--decoder_path', type=str, required=True, help='path for trained decoder')
        self.parser.add_argument('--heatmap_path', type=str, default='', help='path for trained heatmap embedding')
        self.parser.add_argument('--heatmap_trained_path', type=str, required=True, help='path for trained 2D heatmap')
        self.parser.add_argument('--spatial_transformer_path', type=str, default=None, help='path for trained spatial transformer')
        self.parser.add_argument('--heatmap_backbone', type=str, default='convnext_tiny',
                                 help='backbone used by the trained 2D heatmap model')
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples')
        # self.parser.add_argument('--results_dir', type=str, default='./results/', help = 'saves results here')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--save_eval_pose', action='store_true')
        self.parser.add_argument('--save_eval_pose_freq', type=int, default=2000,
                                 help='frequency of saving pose')


        self.isTrain = False
