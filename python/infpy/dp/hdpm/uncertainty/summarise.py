#
# Copyright John Reid 2010
#

"""
Code to summarise inference and output of HDPM.
"""

from cookbook.pylab_utils import pylab_ioff
from cookbook.dicts import DictOf

from ..summarise import heatmap_categories
import logging
import numpy
import os


def threshold_based_on_posterior_enrichment(hdpm, num_topics_used, options):
    """
    Thresholds the posterior of the HDPM to associate TPs with sets of genes and TFs.

    @return: (TP_TF_match, gene_TP_match) - 2 boolean arrays representing the associations
    """
    phi = hdpm.exp_phi()[:num_topics_used]
    Phi = hdpm.exp_Phi()
    phi_ratio = phi / Phi
    topic_factor_match = phi_ratio > options.posterior_enrichment_threshold

    theta = hdpm.exp_theta()[:, :num_topics_used]
    Theta = hdpm.exp_Theta()[:num_topics_used]
    theta_ratio = theta / Theta
    gene_topic_match = theta_ratio > options.posterior_enrichment_threshold

    return gene_topic_match, topic_factor_match


class Statistics(object):
    """
    Calculates statistics about the HDPM.
    """

    def __init__(self, hdpm):
        "Construct."
        self.hdpm = hdpm
        self.update()

    def update(self):
        "Update the statistics."

        self.num_topics_used = (
            self.hdpm.counts.n_k.E > self.hdpm.data.options.topic_size_threshold).sum()
        "The number of topics that have an expected number of factor counts greater than the topic_size_threshold."

        self.gene_topic_match, self.topic_factor_match = threshold_based_on_posterior_enrichment(
            self.hdpm, self.num_topics_used, self.hdpm.data.options
        )

    def log(self, level=logging.INFO):
        for k in range(self.num_topics_used):
            logging.log(level, "Genes for topic % 2d: %s",
                        k, str(self.genes_for_topic(k)))
        for g in range(self.hdpm.data.G):
            logging.log(level, "Topics for gene % 2d: %s",
                        g, str(self.topics_for_gene(g)))
        for k in range(self.num_topics_used):
            logging.log(level, "Factors for topic % 2d: %s",
                        k, str(self.factors_for_topic(k)))
        for f in range(self.hdpm.data.F):
            logging.log(level, "Topics for factor % 2d: %s",
                        f, str(self.topics_for_factor(f)))

    def genes_for_topic(self, k):
        "@return: The indices of those genes that are expected to have drawn from topic k."
        return self.gene_topic_match[:, k].nonzero()[0]

    def topics_for_gene(self, d):
        "@return: The indices of those topics that are expected to have drawn from in gene d."
        return self.gene_topic_match[d].nonzero()[0]

    def factors_for_topic(self, k):
        "@return: The indices of those factors that are likely to be drawn from topic k."
        return self.topic_factor_match[k].nonzero()[0]

    def topics_for_factor(self, w):
        "@return: The indices of those topics that have had factor w drawn from them."
        return self.topic_factor_match[:, w].nonzero()[0]

    def num_topics_per_gene(self):
        "@return: The number of topics per gene."
        return [len(self.topics_for_gene(g)) for g in range(self.hdpm.data.G)]

    def num_genes_per_topic(self):
        "@return: The number of genes per topic."
        return [len(self.genes_for_topic(k)) for k in range(self.num_topics_used)]

    def num_factors_per_topic(self):
        "@return: The number of factors per topic."
        return [len(self.factors_for_topic(k)) for k in range(self.num_topics_used)]

    def num_topics_per_factor(self):
        "@return: The number of topics per factor."
        return [len(self.topics_for_factor(f)) for f in range(self.hdpm.data.F)]

    def genes_for_top_topics(self, k):
        "@return: The genes associated with the topics above index k."
        genes = set()
        for k2 in range(k, self.num_topics_used):
            genes.update(self.genes_for_topic(k2))
        return genes

    def num_genes_by_top_topics(self):
        "@return: The number of genes associated with the top topics."
        return list(map(len, map(self.genes_for_top_topics, range(self.num_topics_used))))


class InferenceHistory(object):
    "Keeps track of statistics during inference."

    def __init__(self, hdpm):

        self.hdpm = hdpm
        "The HDPM."

        self.statistics = Statistics(hdpm)
        "Statistics of the HDPM."

        self.history = DictOf(list)
        "Holds the history of each statistic by name."

        self.stats = {
            'log-likelihood': {
                'log likelihood': lambda: self.hdpm.log_likelihood(),
            },
            'hyper-parameters': {
                'E(alpha)': lambda: self.hdpm.q_alpha.E,
                'G(alpha)': lambda: self.hdpm.q_alpha.G,
                'E(beta)': lambda: self.hdpm.q_beta.E,
                'G(beta)': lambda: self.hdpm.q_beta.G,
                'E(gamma)': lambda: self.hdpm.q_gamma.E,
                'G(gamma)': lambda: self.hdpm.q_gamma.G,
            },
            'topic-sizes': {
                'E(factors/topic)': lambda: numpy.mean(self.statistics.num_factors_per_topic()),
                'E(topics/factor)': lambda: numpy.mean(self.statistics.num_topics_per_factor()),
                'E(topics/gene)': lambda: numpy.mean(self.statistics.num_topics_per_gene()),
                'E(genes/topic)': lambda: numpy.mean(self.statistics.num_genes_per_topic()),
            },
            'num_topics_used': {
                '# topics used': lambda: self.statistics.num_topics_used,
            },
        }
        "A dictionary of dictionaries. Top level keys are plot names, second level keys are stats in that plot."

    def update(self):
        for _plot, stats in self.stats.items():
            for name, fn in stats.items():
                self.history[name].append(fn())
        return self.history['log likelihood'][-1]

    @pylab_ioff
    def make_plots(self, directory, formats=('png',)):
        import pylab as P
        for plot, stats in self.stats.items():
            P.figure()
            for name, _fn in stats.items():
                P.plot(self.history[name], label=name)
            P.legend(loc='upper left')
            for format in formats:
                P.savefig(os.path.join(directory, 'history-%s.%s') %
                          (plot, format))
            P.close()


class Summariser(object):
    """
    Summarises HDPM.
    """

    def __init__(
        self,
        hdpm,
        filename_prefix,
        gene_ids=None,
        factor_ids=None,
        gene_tag='gene',
        topic_tag='topic',
        factor_tag='factor',
        site_tag='site'
    ):
        """
        Constructs a DpmSummariser.

        @arg dpm: The Dirichlet process mixture.
        @arg filename_prefix: A prefix for the filenames the summariser generates.
        @arg gene_ids: A list of identifiers for the genes.
        @arg factor_ids: A list of identifiers for the factors.
        @arg gene_tag: A string to replace the word "gene" with in the output.
        @arg topic_tag: A string to replace the word "topic" with in the output.
        @arg factor_tag: A string to replace the word "factor" with in the output.
        @arg site_tag: A string to replace the word "occurence" with in the output.
        """

        if None == gene_ids:
            gene_ids = [str(i) for i in range(hdpm.data.G)]
        if None == factor_ids:
            factor_ids = [str(i) for i in range(hdpm.data.F)]

        self.hdpm = hdpm
        "The Dirichlet process mixture."

        self.statistics = Statistics(hdpm)
        "Statistics about the DPM."

        self.prefix = filename_prefix
        "A prefix for the filenames the summariser generates."

        self.gene_ids = gene_ids
        "A list of identifiers for the genes."

        self.factor_ids = factor_ids
        "A list of identifiers for the factors."

        self.gene_tag = gene_tag
        'A string to replace the factor "gene" with in the output.'

        self.topic_tag = topic_tag
        'A string to replace the factor "topic" with in the output.'

        self.factor_tag = factor_tag
        'A string to replace the factor "factor" with in the output.'

        self.site_tag = site_tag
        'A string to replace the factor "occurence" with in the output.'

    def summarise_all(self):
        self.log_static_info()
        self.log_dynamic_info()
        self.log_hyper_parameter_info()

        self.plot_factor_gene_scatter()
        self.plot_num_genes_by_top_topics()
        self.make_heat_maps()
        self.topic_sizes()
        self.histograms()

        for k in range(self.statistics.num_topics_used):
            self.log_topic_info(k)
            self.plot_factor_enrichment(k)

    def log_static_info(self):
        'Log information about the DPM that does not change.'
        logging.info('DPM data has %d %ss', self.hdpm.data.G, self.gene_tag)
        logging.info('DPM data has %d %ss', self.hdpm.data.F, self.factor_tag)
        logging.info('DPM is restricted to %d %ss',
                     self.hdpm.K, self.topic_tag)
        logging.info('DPM data has %d %ss',
                     self.hdpm.data.n_g.sum(), self.site_tag)

    def log_dynamic_info(self):
        'Log information about the DPM that does change.'

        self.statistics.update()

        logging.info('Using %d %ss',
                     self.statistics.num_topics_used, self.topic_tag)

        logging.info(
            'Average number %ss per %s = %f',
            self.topic_tag,
            self.factor_tag,
            numpy.mean(self.statistics.num_topics_per_factor())
        )

        logging.info(
            'Average number %ss per %s = %f',
            self.factor_tag,
            self.topic_tag,
            numpy.mean(self.statistics.num_factors_per_topic())
        )

        logging.info(
            'Average number %ss per %s = %f',
            self.gene_tag,
            self.topic_tag,
            numpy.mean(self.statistics.num_genes_per_topic())
        )

        logging.info(
            'Average number %ss per %s = %f',
            self.topic_tag,
            self.gene_tag,
            numpy.mean(self.statistics.num_topics_per_gene())
        )

        self.log_hyper_parameter_info()

    @pylab_ioff
    def save_fig(self, tag, **kwargs):
        from pylab import savefig
        #savefig('%s-%s.eps' % (self.prefix, tag), format='EPS', **kwargs)
        savefig('%s-%s.png' % (self.prefix, tag), format='PNG', **kwargs)

    @pylab_ioff
    def pcolor(self, Z, tag, size=1, **kwargs):
        import pylab as P
        P.figure()
        P.pcolor(Z)
        P.gca().set_xlim((0, Z.shape[0]))
        P.gca().set_ylim((0, Z.shape[1]))
        P.colorbar()
        self.save_fig(tag)
        P.close()

    @pylab_ioff
    def imshow(self, Z, tag, size=1, **kwargs):
        import pylab as P
        dpi = 100.
        figsize = numpy.array(Z.shape) / dpi
        # P.rcParams.update({'figure.figsize':figsize})
        fig = P.figure(figsize=figsize)
        P.axes([0, 0, 1, 1])  # Make the plot occupy the whole canvas
        P.axis('off')
        fig.set_size_inches(size * figsize)
        P.imshow(Z, origin='lower', **kwargs)
        self.save_fig(tag, facecolor='black', edgecolor='black', dpi=dpi)
        P.close()

    @pylab_ioff
    def plot_factor_gene_scatter(self):
        "Make a scatter plot of number of factors against number of genes for each topic."
        logging.info('Plotting factor gene scatter')
        import pylab as P
        P.figure()
        sizes = 3000. * self.hdpm.counts.n_k.E / self.hdpm.data.N
        P.scatter(
            self.statistics.num_factors_per_topic(),
            self.statistics.num_genes_per_topic(),
            s=sizes
        )
        P.xlim(xmin=0)
        P.ylim(ymin=0)
        P.xlabel('# %ss' % self.factor_tag)
        P.ylabel('# %ss' % self.gene_tag)
        P.title('%ss vs. %ss vs. %ss' %
                (self.factor_tag, self.gene_tag, self.site_tag))
        self.save_fig('%s-%s-scatter' % (self.factor_tag, self.gene_tag))
        P.close()

    @pylab_ioff
    def plot_num_genes_by_top_topics(self):
        "Make a plot of number of genes depending on number of topics included."
        logging.info('Plotting number of genes')
        import pylab as P
        P.figure()
        P.plot(self.statistics.num_genes_by_top_topics())
        P.xlabel('%s cut-off' % self.topic_tag)
        P.ylabel('# %ss' % self.gene_tag)
        # P.title('# %ss associated with top programs by size' % self.gene_tag)
        self.save_fig('num-%ss-by-top-%ss' % (self.gene_tag, self.topic_tag))
        P.close()

    def make_topic_KL_heat_map(self):
        "Make a heat map representing the distances between transcriptional programs."
        phi = self.hdpm.exp_phi()
        K = self.statistics.num_topics_used
        Z = numpy.empty((K, K))
        for k1 in range(K):
            for k2 in range(K):
                Z[k1, k2] = discrete_KL(phi[k1], phi[k2])
        #self.imshow(Z, '%s-KL' % self.topic_tag, size=20, interpolation='nearest')
        self.pcolor(Z, '%s-KL' % self.topic_tag)
        return Z

    def make_topic_factor_intersection_heat_map(self):
        "Make a heat map representing the distances between transcriptional programs."
        K = self.statistics.num_topics_used
        factors = [set(self.statistics.factors_for_topic(k))
                   for k in range(K)]
        Z = numpy.zeros((K, K))
        for k1 in range(K):
            for k2 in range(K):
                if len(factors[k2]):
                    Z[k1, k2] = 1. - \
                        len(factors[k1].intersection(factors[k2])
                            ) / float(len(factors[k2]))
        #self.imshow(Z, '%s-%s-intersections' % (self.topic_tag, self.factor_tag), size=20, interpolation='nearest')
        self.pcolor(Z, '%s-%s-intersections' %
                    (self.topic_tag, self.factor_tag))
        return Z

    def make_topic_gene_intersection_heat_map(self):
        "Make a heat map representing the distances between transcriptional programs."
        K = self.statistics.num_topics_used
        genes = [set(self.statistics.genes_for_topic(k)) for k in range(K)]
        Z = numpy.zeros((K, K))
        for k1 in range(K):
            for k2 in range(K):
                if len(genes[k2]):
                    Z[k1, k2] = 1. - \
                        len(genes[k1].intersection(genes[k2])) / \
                        float(len(genes[k2]))
        #self.imshow(Z, '%s-%s-intersections' % (self.topic_tag, self.gene_tag), size=20, interpolation='nearest')
        self.pcolor(Z, '%s-%s-intersections' % (self.topic_tag, self.gene_tag))
        return Z

    @pylab_ioff
    def make_phi_heat_map(self):
        "Make a heat map representing the expected phis."
        K = self.statistics.num_topics_used
        import pylab as P
        P.figure()
        heatmap_categories(self.hdpm.exp_phi()[
                           :K], category_names=self.factor_ids, distribution_names=list(map(str, list(range(K)))))
        self.save_fig('expected-phi')
        P.close()

    @pylab_ioff
    def make_theta_heat_map(self):
        "Make a heat map representing the expected thetas."
        K = self.statistics.num_topics_used
        import pylab as P
        P.figure()
        if len(self.gene_ids) < 30:
            gene_ids = self.gene_ids
        else:
            gene_ids = None
        heatmap_categories(self.hdpm.exp_theta()[:, :K], category_names=list(map(
            str, list(range(K)))), distribution_names=gene_ids)
        self.save_fig('expected-theta')
        P.close()

    def make_heat_maps(self):
        # self.make_topic_KL_heat_map()
        logging.info('Creating topic intersection heat maps')
        self.make_phi_heat_map()
        # self.make_theta_heat_map() normally too many genes to make this sensible
        self.make_topic_factor_intersection_heat_map()
        self.make_topic_gene_intersection_heat_map()

    @pylab_ioff
    def topic_sizes(self):
        'Create a PNG of the topic sizes, i.e. the expected number of occurences from each topic.'
        logging.info('Creating topic size histogram')
        import pylab as P
        P.figure()
        P.bar(
            numpy.arange(self.statistics.num_topics_used) - .4,
            self.hdpm.counts.n_k.E[:self.statistics.num_topics_used],
            width=.8,
        )
        P.xlim(-.5, self.statistics.num_topics_used - .5)
        P.title('%s sizes' % self.topic_tag)
        P.xlabel('Programs')
        P.ylabel('Sizes')
        self.save_fig('%s-sizes' % self.topic_tag)
        P.close()

    @pylab_ioff
    def topic_sizes_log_scale(self):
        'Create a PNG of the topic sizes, i.e. the expected number of occurences from each topic.'
        logging.info('Creating topic size histogram (log scale)')
        import pylab as P
        P.figure()
        P.bar(
            numpy.arange(self.statistics.num_topics_used) - .4,
            self.hdpm.counts.n_k.E[:self.statistics.num_topics_used],
            width=.8,
            log=True
        )
        P.xlim(-.5, self.statistics.num_topics_used - .5)
        P.title('%s sizes (log scale)' % self.topic_tag)
        P.xlabel('Programs')
        P.ylabel('Sizes')
        self.save_fig('%s-sizes-log-scale' % self.topic_tag)
        P.close()

    @pylab_ioff
    def _histogram(self, data, count_tag, index_tag):
        import pylab as P
        P.figure()
        xticks = numpy.arange(min(data), max(data) + 2)
        bins = xticks - .5
        P.hist(numpy.asarray(data), bins, rwidth=.8)
        P.title('Number of %ss associated with each %s' %
                (count_tag, index_tag))
        P.xlabel('# %ss' % count_tag)
        P.ylabel('# %ss' % index_tag)
        # P.xticks(xticks[:-1])
        self.save_fig('hist-%s-per-%s' % (count_tag, index_tag))
        P.close()

    def histograms(self):
        'Create PNG histograms of the various statistics.'
        logging.info('Creating factor/topic/gene histograms')
        self._histogram(
            self.statistics.num_topics_per_gene(),
            self.topic_tag,
            self.gene_tag
        )
        self._histogram(
            self.statistics.num_genes_per_topic(),
            self.gene_tag,
            self.topic_tag
        )
        self._histogram(
            self.statistics.num_factors_per_topic(),
            self.factor_tag,
            self.topic_tag
        )
        self._histogram(
            self.statistics.num_topics_per_factor(),
            self.topic_tag,
            self.factor_tag
        )

    @pylab_ioff
    def _plot_dist(self, dist, title, tag):
        import pylab as P
        P.figure()
        dist.plot()
        self.save_fig(tag)
        P.close()

    def _dist_plots(self):
        self._plot_dist(self.hdpm.q_alpha, 'Alpha', 'alpha')
        self._plot_dist(self.hdpm.q_beta, 'Beta', 'beta')
        self._plot_dist(self.hdpm.q_gamma, 'Gamma', 'gamma')

    def _log_hyper_info(self, p, name):
        logging.info('%8s: E=%4g; G=%4g; params: %s',
                     name, p.E, p.G, str(p.params()))

    def log_hyper_parameter_info(self):
        'Log some information about the distributions over the hyper-parameters.'
        self._log_hyper_info(self.hdpm.q_alpha, 'Alpha')
        self._log_hyper_info(self.hdpm.q_beta, 'Beta')
        self._log_hyper_info(self.hdpm.q_gamma, 'Gamma')

    def log_topic_info(self, k):
        'Log some general information about topic k.'
        topic_factors = self.statistics.factors_for_topic(k)
        topic_genes = self.statistics.genes_for_topic(k)
        logging.info(
            '%s: ********************************** %d **********************************', self.topic_tag, k)
        logging.info('Expected # %ss count: %f',
                     self.site_tag, self.hdpm.counts.n_k.E[k])
        logging.info('Number distinct %ss: %d',
                     self.factor_tag, len(topic_factors))
        logging.info('%ss: %s', self.factor_tag, ', '.join(
            self.factor_ids[w] for w in topic_factors))
        logging.info('Number %ss using topic: %d',
                     self.gene_tag, len(topic_genes))
        logging.info('%ss: %s', self.gene_tag, ', '.join(
            self.gene_ids[d] for d in topic_genes))

    @pylab_ioff
    def plot_factor_enrichment(self, k):
        'Plots the enrichment this program has for the various factors.'
        import pylab as P
        fig = P.figure()
        P.figtext(0.03, 0.97, 'Factor enrichment in program %d: %d %ss' % (
            k, len(self.statistics.factors_for_topic(k)), self.factor_tag))

        P.subplot(211)
        P.bar(list(range(self.hdpm.data.F)), self.hdpm.counts.n_kf.E[k])
        P.title('Expected number %ss by %s.' %
                (self.site_tag, self.factor_tag))

        P.subplot(212)
        phi = self.hdpm.exp_phi()[:self.statistics.num_topics_used]
        Phi = self.hdpm.exp_Phi()
        phi_ratio = phi / Phi
        P.bar(list(range(self.hdpm.data.F)), phi_ratio[k])
        P.title('Enrichment ratio over background %s distribution' %
                self.factor_tag)

        return fig
