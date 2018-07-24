import numpy as np
from .other import clip_boxes
from .text_proposal_graph_builder import TextProposalGraphBuilder

class TextProposalConnector:
    def __init__(self):
        self.graph_builder=TextProposalGraphBuilder()
    #将提议区域放到一个组中。
    def group_text_proposals(self, text_proposals, scores, im_size):
        graph=self.graph_builder.build_graph(text_proposals, scores, im_size)
        return graph.sub_graphs_connected()
    #拟合边框线的两端y坐标
    def fit_y(self, X, Y, x1, x2):
        len(X)!=0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X==X[0])==len(X):
            return Y[0], Y[0]
        p=np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        #print(text_proposals)
        #print(scores)
        # tp=text proposal
        tp_groups=self.group_text_proposals(text_proposals, scores, im_size)
        text_lines=np.zeros((len(tp_groups), 5), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes=text_proposals[list(tp_indices)]

            x0=np.min(text_line_boxes[:, 0])
            x1=np.max(text_line_boxes[:, 2])
            #一个提议框的一半宽度
            offset=(text_line_boxes[0, 2]-text_line_boxes[0, 0])*0.5
            #上下两条线的两端y坐标
            lt_y, rt_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0+offset, x1-offset)
            lb_y, rb_y=self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0+offset, x1-offset)

            # the score of a text line is the average score of the scores
            # of all text proposals contained in the text line
            score=scores[list(tp_indices)].sum()/float(len(tp_indices))
            #标定每个提议组的框的范围
            text_lines[index, 0]=x0
            text_lines[index, 1]=min(lt_y, rt_y)
            text_lines[index, 2]=x1
            text_lines[index, 3]=max(lb_y, rb_y)
            text_lines[index, 4]=score
        #去掉超出图片范围的线
        text_lines=clip_boxes(text_lines, im_size)
        
        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        #标出每个框的四个顶点坐标
        for line in text_lines:
            xmin,ymin,xmax,ymax=line[0],line[1],line[2],line[3]
            text_recs[index, 0] = xmin
            text_recs[index, 1] = ymin
            text_recs[index, 2] = xmax
            text_recs[index, 3] = ymax
            text_recs[index, 8] = line[4]
            index = index + 1

        return text_recs
