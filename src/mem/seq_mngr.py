import numpy as np
from src.utils.text_sty import print_info, print_warning


class SequenceManager:
    """Helper class for ReplayMemory. Keeps track of sequences in real-time."""

    START_PTRS = 0
    ENV_IDS = 1
    LENGTHS = 2
    PRIOS = 3

    def __init__(self, buffer_len, par_inst, seq_len, seq_shift, eta):
        self.buffer_len = buffer_len
        self.par_inst = par_inst
        self.seq_len = seq_len
        self.seq_shift = seq_shift
        self.seq_overlap = seq_len - seq_shift
        self.eta = eta

        self.sequences = np.zeros(shape=(self.buffer_len, 4), dtype='float32')
        self.seq_len_cntr = np.zeros(self.par_inst)
        self.running_seq_starts = np.zeros(self.par_inst)
        self.stack_ptr = 0

    def update(self, terminals, trans_stack_ptr, init_prio):
        # TODO: forgot to take seq_overlap into account!
        self.seq_len_cntr += 1
        seq_end = terminals | self.seq_len_cntr == self.seq_len
        num_end_seq = np.sum(seq_end)

        if num_end_seq > 0:
            remaining_space = self.buffer_len - self.stack_ptr
            if num_end_seq > remaining_space:
                raise RuntimeError("SequenceManager full!")

            seq_end_ids = np.where(seq_end)
            fin_env_ids = np.where(terminals)
            start_ptrs = self.running_seq_starts[seq_end_ids]
            lengths = self.seq_len_cntr[seq_end_ids]
            prios = np.ones(num_end_seq) * init_prio
            new_seqs = np.stack([fin_env_ids, start_ptrs, lengths, prios])
            self.sequences[self.stack_ptr:self.stack_ptr + num_end_seq] = new_seqs
            self.seq_len_cntr[seq_end] = 0

            if self.stack_ptr / self.buffer_len > 0.98:
                print_info("Info: SequenceManager is running out of space! Only %.1f %% (%d sequence spots) left!" %
                           (100 - self.stack_ptr / self.buffer_len * 100, remaining_space))

        self.stack_ptr += num_end_seq
        self.running_seq_starts[seq_end] = trans_stack_ptr[seq_end]

    def get_num_seqs(self):
        return self.stack_ptr

    def get_seq_prios(self):
        return self.sequences[:self.stack_ptr, self.PRIOS]

    def seq_ids_to_trans_ids(self, seq_ids, max_trans_idx):
        num_seqs = len(seq_ids)

        seq_starts = self.sequences[seq_ids, 0]
        seq_lengths = self.sequences[seq_ids, 1]
        seq_max_ends = seq_starts + self.seq_len

        trans_ids = np.stack([np.arange(start, end) for start, end in zip(seq_starts, seq_max_ends)], axis=0)
        trans_ids[trans_ids >= max_trans_idx] = 0  # are masked away anyway, but ids need to be legal indices

        # Mask sequences accordingly
        mask = np.zeros(shape=(num_seqs, self.seq_len), dtype='bool')
        for seq in range(num_seqs):
            mask[seq, 0:seq_lengths[seq]] = True

        return trans_ids, mask

    def update_seq_priorities(self, seq_ids, trans_ids=None, mask=None):
        if trans_ids is None:
            trans_ids, mask = self.seq_ids_to_trans_ids(seq_ids)  # TODO
        priorities = self.get_seq_prios()[trans_ids].copy()
        priorities[~ mask] = 0
        prio_max = np.max(priorities, axis=1)
        prio_avg = np.average(priorities, axis=1)
        seq_prios = self.eta * prio_max + (1 - self.eta) * prio_avg
        self.sequences[seq_ids, self.PRIOS] = seq_prios
        return seq_prios

    def delete_first(self, n):
        """Deletes sequences starting at one of the first n transitions. Keeps the data in-place."""
        assert n <= self.stack_ptr
        to_keep = self.sequences[:self.stack_ptr, self.START_PTRS] >= n
        num_to_keep = np.sum(to_keep)
        new_sequence_lst = self.sequences[:self.stack_ptr][to_keep]
        self.sequences[:num_to_keep] = new_sequence_lst
        self.sequences[num_to_keep:] = 0
        self.stack_ptr -= np.sum(~ to_keep)
