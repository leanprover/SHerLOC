/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Constants
import SHerLOC.AST.Identifiers
import SHerLOC.AST.Types

/-!
# Operations

-/

namespace StableHLO

inductive OpName where
  | abs
  | add
  | after_all
  | all_gather
  | all_reduce
  | all_to_all
  | and
  | atan2
  | batch_norm_grad
  | batch_norm_inference
  | batch_norm_training
  | bitcast_convert
  | broadcast_in_dim
  | case
  | cbrt
  | ceil
  | cholesky
  | clamp
  | collective_broadcast
  | collective_permute
  | compare
  | complex
  | composite
  | concatenate
  | constant
  | convert
  | convolution
  | cosine
  | count_leading_zeros
  | custom_call
  | divide
  | dot_general
  | dynamic_broadcast_in_dim
  | dynamic_conv
  | dynamic_gather
  | dynamic_iota
  | dynamic_pad
  | dynamic_reshape
  | dynamic_slice
  | dynamic_update_slice
  | exponential
  | exponential_minus_one
  | fft
  | floor
  | gather
  | get_dimension_size
  | get_tuple_element
  | if
  | imag
  | infeed
  | iota
  | is_finite
  | log
  | log_plus_one
  | logistic
  | map
  | maximum
  | minimum
  | multiply
  | negate
  | not
  | optimization_barrier
  | or
  | outfeed
  | pad
  | partition_id
  | popcnt
  | power
  | real
  | recv
  | reduce
  | reduce_precision
  | reduce_scatter
  | reduce_window
  | remainder
  | replica_id
  | reshape
  | reverse
  | rng
  | rng_bit_generator
  | round_nearest_afz
  | round_nearest_even
  | rsqrt
  | scatter
  | select
  | select_and_scatter
  | send
  | shift_left
  | shift_right_arithmetic
  | shift_right_logical
  | sign
  | sine
  | slice
  | sort
  | sqrt
  | subtract
  | tan
  | tanh
  | transpose
  | triangular_solve
  | tuple
  | uniform_dequantize
  | uniform_quantize
  | while
  | xor
  deriving Repr, Inhabited, Nonempty

  structure Attribute where
    id : AttrId
    constant : Constant
    deriving Repr, Inhabited, Nonempty

structure FuncInput where
  id : FuncId
  typ : ValueType
  deriving Repr, Inhabited, Nonempty

mutual

  inductive InputFunc where
    | mk
      (id : UnusedId)
      (funcInputs : List FuncInput)
      (body : List Operation)
    deriving Repr, Inhabited, Nonempty

  inductive Operation where
    | stable
      (name : OpName)(inputValues : List ValueId)
      (inputFunctions : List InputFunc)
      (inputAttributes : List Attribute)
      (outputs : List ValueId)
      (signature : FunctionType)
    | return
      (operands : List ValueId)
      (signature : FunctionType)
    | constant
      (outputs : List ValueId)
      (value : Constant)
    deriving Repr, Inhabited, Nonempty

end

end StableHLO
