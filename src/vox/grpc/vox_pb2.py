




"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import runtime_version as _runtime_version
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
_runtime_version.ValidateProtobufRuntimeVersion(
    _runtime_version.Domain.PUBLIC,
    6,
    31,
    1,
    '',
    'vox.proto'
)


_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\tvox.proto\x12\x03vox\"\x0f\n\rHealthRequest\" \n\x0eHealthResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\"\x13\n\x11ListLoadedRequest\"6\n\x12ListLoadedResponse\x12 \n\x06models\x18\x01 \x03(\x0b\x32\x10.vox.LoadedModel\"\x93\x01\n\x0bLoadedModel\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0b\n\x03tag\x18\x02 \x01(\t\x12\x0c\n\x04type\x18\x03 \x01(\t\x12\x0e\n\x06\x64\x65vice\x18\x04 \x01(\t\x12\x12\n\nvram_bytes\x18\x05 \x01(\x03\x12\x11\n\tloaded_at\x18\x06 \x01(\x01\x12\x11\n\tlast_used\x18\x07 \x01(\x01\x12\x11\n\tref_count\x18\x08 \x01(\x05\"\x1b\n\x0bPullRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\"O\n\x0cPullProgress\x12\x0e\n\x06status\x18\x01 \x01(\t\x12\r\n\x05\x65rror\x18\x02 \x01(\t\x12\x11\n\tcompleted\x18\x03 \x01(\x05\x12\r\n\x05total\x18\x04 \x01(\x05\"\x13\n\x11ListModelsRequest\"4\n\x12ListModelsResponse\x12\x1e\n\x06models\x18\x01 \x03(\x0b\x32\x0e.vox.ModelInfo\"v\n\tModelInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\x0e\n\x06\x66ormat\x18\x03 \x01(\t\x12\x14\n\x0c\x61rchitecture\x18\x04 \x01(\t\x12\x12\n\nsize_bytes\x18\x05 \x01(\x03\x12\x13\n\x0b\x64\x65scription\x18\x06 \x01(\t\"\x1b\n\x0bShowRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\"\x9a\x01\n\x0cShowResponse\x12\x0c\n\x04name\x18\x01 \x01(\t\x12-\n\x06\x63onfig\x18\x02 \x03(\x0b\x32\x1d.vox.ShowResponse.ConfigEntry\x12\x1e\n\x06layers\x18\x03 \x03(\x0b\x32\x0e.vox.LayerInfo\x1a-\n\x0b\x43onfigEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\"O\n\tLayerInfo\x12\x12\n\nmedia_type\x18\x01 \x01(\t\x12\x0e\n\x06\x64igest\x18\x02 \x01(\t\x12\x0c\n\x04size\x18\x03 \x01(\x03\x12\x10\n\x08\x66ilename\x18\x04 \x01(\t\"\x1d\n\rDeleteRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\" \n\x0e\x44\x65leteResponse\x12\x0e\n\x06status\x18\x01 \x01(\t\"\x9f\x01\n\x11TranscribeRequest\x12\r\n\x05\x61udio\x18\x01 \x01(\x0c\x12\r\n\x05model\x18\x02 \x01(\t\x12\x10\n\x08language\x18\x03 \x01(\t\x12\x17\n\x0fword_timestamps\x18\x04 \x01(\x08\x12\x13\n\x0btemperature\x18\x05 \x01(\x02\x12\x17\n\x0fresponse_format\x18\x06 \x01(\t\x12\x13\n\x0b\x66ormat_hint\x18\x07 \x01(\t\"\xc8\x01\n\x12TranscribeResponse\x12\r\n\x05model\x18\x01 \x01(\t\x12\x0c\n\x04text\x18\x02 \x01(\t\x12\x10\n\x08language\x18\x03 \x01(\t\x12\x13\n\x0b\x64uration_ms\x18\x04 \x01(\x05\x12\x15\n\rprocessing_ms\x18\x05 \x01(\x05\x12(\n\x08segments\x18\x06 \x03(\x0b\x32\x16.vox.TranscriptSegment\x12\x1d\n\x08\x65ntities\x18\x07 \x03(\x0b\x32\x0b.vox.Entity\x12\x0e\n\x06topics\x18\x08 \x03(\t\"J\n\x06\x45ntity\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x0c\n\x04text\x18\x02 \x01(\t\x12\x12\n\nstart_char\x18\x03 \x01(\r\x12\x10\n\x08\x65nd_char\x18\x04 \x01(\r\"1\n\x0f\x41nnotateRequest\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x10\n\x08language\x18\x02 \x01(\t\"A\n\x10\x41nnotateResponse\x12\x1d\n\x08\x65ntities\x18\x01 \x03(\x0b\x32\x0b.vox.Entity\x12\x0e\n\x06topics\x18\x02 \x03(\t\"f\n\x11TranscriptSegment\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x10\n\x08start_ms\x18\x02 \x01(\x05\x12\x0e\n\x06\x65nd_ms\x18\x03 \x01(\x05\x12!\n\x05words\x18\x04 \x03(\x0b\x32\x12.vox.WordTimestamp\"g\n\rWordTimestamp\x12\x0c\n\x04word\x18\x01 \x01(\t\x12\x10\n\x08start_ms\x18\x02 \x01(\x05\x12\x0e\n\x06\x65nd_ms\x18\x03 \x01(\x05\x12\x17\n\nconfidence\x18\x04 \x01(\x02H\x00\x88\x01\x01\x42\r\n\x0b_confidence\"z\n\x11SynthesizeRequest\x12\r\n\x05model\x18\x01 \x01(\t\x12\r\n\x05input\x18\x02 \x01(\t\x12\r\n\x05voice\x18\x03 \x01(\t\x12\r\n\x05speed\x18\x04 \x01(\x02\x12\x10\n\x08language\x18\x05 \x01(\t\x12\x17\n\x0fresponse_format\x18\x06 \x01(\t\"B\n\nAudioChunk\x12\r\n\x05\x61udio\x18\x01 \x01(\x0c\x12\x13\n\x0bsample_rate\x18\x02 \x01(\x05\x12\x10\n\x08is_final\x18\x03 \x01(\x08\"\xdd\x01\n\x0bStreamInput\x12#\n\x06\x63onfig\x18\x01 \x01(\x0b\x32\x11.vox.StreamConfigH\x00\x12 \n\x05\x61udio\x18\x02 \x01(\x0b\x32\x0f.vox.AudioFrameH\x00\x12$\n\nopus_frame\x18\x03 \x01(\x0b\x32\x0e.vox.OpusFrameH\x00\x12/\n\rencoded_audio\x18\x04 \x01(\x0b\x32\x16.vox.EncodedAudioFrameH\x00\x12)\n\rend_of_stream\x18\x05 \x01(\x0b\x32\x10.vox.EndOfStreamH\x00\x42\x05\n\x03msg\"\r\n\x0b\x45ndOfStream\"\xc2\x01\n\x0cStreamConfig\x12\x10\n\x08language\x18\x01 \x01(\t\x12\x13\n\x0bsample_rate\x18\x02 \x01(\r\x12\r\n\x05model\x18\x03 \x01(\t\x12\x10\n\x08partials\x18\x04 \x01(\x08\x12\x19\n\x11partial_window_ms\x18\x05 \x01(\r\x12\x19\n\x11partial_stride_ms\x18\x06 \x01(\r\x12\x1f\n\x17include_word_timestamps\x18\x07 \x01(\x08\x12\x13\n\x0btemperature\x18\x08 \x01(\x02\"0\n\nAudioFrame\x12\r\n\x05pcm16\x18\x01 \x01(\x0c\x12\x13\n\x0bsample_rate\x18\x02 \x01(\r\"@\n\tOpusFrame\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\x13\n\x0bsample_rate\x18\x02 \x01(\r\x12\x10\n\x08\x63hannels\x18\x03 \x01(\r\"1\n\x11\x45ncodedAudioFrame\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x12\x0e\n\x06\x66ormat\x18\x02 \x01(\t\"\xfd\x01\n\x0cStreamOutput\x12!\n\x05ready\x18\x01 \x01(\x0b\x32\x10.vox.StreamReadyH\x00\x12\x32\n\x0espeech_started\x18\x02 \x01(\x0b\x32\x18.vox.StreamSpeechStartedH\x00\x12\x32\n\x0espeech_stopped\x18\x03 \x01(\x0b\x32\x18.vox.StreamSpeechStoppedH\x00\x12\x31\n\ntranscript\x18\x04 \x01(\x0b\x32\x1b.vox.StreamTranscriptResultH\x00\x12(\n\x05\x65rror\x18\x05 \x01(\x0b\x32\x17.vox.StreamErrorMessageH\x00\x42\x05\n\x03msg\"\r\n\x0bStreamReady\"+\n\x13StreamSpeechStarted\x12\x14\n\x0ctimestamp_ms\x18\x01 \x01(\r\"+\n\x13StreamSpeechStopped\x12\x14\n\x0ctimestamp_ms\x18\x01 \x01(\r\"\xd4\x02\n\x16StreamTranscriptResult\x12\x0c\n\x04text\x18\x01 \x01(\t\x12\x12\n\nis_partial\x18\x02 \x01(\x08\x12\x10\n\x08start_ms\x18\x03 \x01(\r\x12\x0e\n\x06\x65nd_ms\x18\x04 \x01(\r\x12\x19\n\x11\x61udio_duration_ms\x18\x05 \x01(\r\x12\x1e\n\x16processing_duration_ms\x18\x06 \x01(\r\x12\r\n\x05model\x18\x07 \x01(\t\x12\x1c\n\x0f\x65ou_probability\x18\x08 \x01(\x02H\x00\x88\x01\x01\x12\x1d\n\x08\x65ntities\x18\t \x03(\x0b\x32\x0b.vox.Entity\x12\x0e\n\x06topics\x18\n \x03(\t\x12!\n\x05words\x18\x0b \x03(\x0b\x32\x12.vox.WordTimestamp\x12(\n\x08segments\x18\x0c \x03(\x0b\x32\x16.vox.TranscriptSegmentB\x12\n\x10_eou_probability\"%\n\x12StreamErrorMessage\x12\x0f\n\x07message\x18\x01 \x01(\t\"\"\n\x11ListVoicesRequest\x12\r\n\x05model\x18\x01 \x01(\t\"4\n\x12ListVoicesResponse\x12\x1e\n\x06voices\x18\x01 \x03(\x0b\x32\x0e.vox.VoiceInfo\"\x80\x01\n\x12\x43reateVoiceRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05\x61udio\x18\x02 \x01(\x0c\x12\x10\n\x08language\x18\x03 \x01(\t\x12\x0e\n\x06gender\x18\x04 \x01(\t\x12\x16\n\x0ereference_text\x18\x05 \x01(\t\x12\x13\n\x0b\x66ormat_hint\x18\x06 \x01(\t\"H\n\x13\x43reateVoiceResponse\x12\x1d\n\x05voice\x18\x01 \x01(\x0b\x32\x0e.vox.VoiceInfo\x12\x12\n\ncreated_at\x18\x02 \x01(\x03\" \n\x12\x44\x65leteVoiceRequest\x12\n\n\x02id\x18\x01 \x01(\t\"2\n\x13\x44\x65leteVoiceResponse\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0f\n\x07\x64\x65leted\x18\x02 \x01(\x08\"~\n\tVoiceInfo\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x10\n\x08language\x18\x03 \x01(\t\x12\x0e\n\x06gender\x18\x04 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x05 \x01(\t\x12\x11\n\tis_cloned\x18\x06 \x01(\x08\x12\r\n\x05model\x18\x07 \x01(\t\"\x91\x03\n\x15\x43onverseClientMessage\x12\x38\n\x0esession_update\x18\x01 \x01(\x0b\x32\x1e.vox.ConversationSessionUpdateH\x00\x12\x34\n\x0c\x61udio_append\x18\x02 \x01(\x0b\x32\x1c.vox.ConversationAudioAppendH\x00\x12:\n\x0fresponse_cancel\x18\x04 \x01(\x0b\x32\x1f.vox.ConversationResponseCancelH\x00\x12\x38\n\x0eresponse_start\x18\x05 \x01(\x0b\x32\x1e.vox.ConversationResponseStartH\x00\x12\x38\n\x0eresponse_delta\x18\x06 \x01(\x0b\x32\x1e.vox.ConversationResponseDeltaH\x00\x12:\n\x0fresponse_commit\x18\x07 \x01(\x0b\x32\x1f.vox.ConversationResponseCommitH\x00\x42\x05\n\x03msgJ\x04\x08\x03\x10\x04R\x0fresponse_create\"\x99\x05\n\x15\x43onverseServerMessage\x12:\n\x0fsession_created\x18\x01 \x01(\x0b\x32\x1f.vox.ConversationSessionCreatedH\x00\x12\x38\n\x0espeech_started\x18\x02 \x01(\x0b\x32\x1e.vox.ConversationSpeechStartedH\x00\x12\x38\n\x0espeech_stopped\x18\x03 \x01(\x0b\x32\x1e.vox.ConversationSpeechStoppedH\x00\x12:\n\x0ftranscript_done\x18\x04 \x01(\x0b\x32\x1f.vox.ConversationTranscriptDoneH\x00\x12<\n\x10response_created\x18\x05 \x01(\x0b\x32 .vox.ConversationResponseCreatedH\x00\x12\x32\n\x0b\x61udio_delta\x18\x06 \x01(\x0b\x32\x1b.vox.ConversationAudioDeltaH\x00\x12\x36\n\rresponse_done\x18\x07 \x01(\x0b\x32\x1d.vox.ConversationResponseDoneH\x00\x12@\n\x12response_cancelled\x18\x08 \x01(\x0b\x32\".vox.ConversationResponseCancelledH\x00\x12\x36\n\rstate_changed\x18\t \x01(\x0b\x32\x1d.vox.ConversationStateChangedH\x00\x12\'\n\x05\x65rror\x18\n \x01(\x0b\x32\x16.vox.ConversationErrorH\x00\x12@\n\x12response_committed\x18\x0b \x01(\x0b\x32\".vox.ConversationResponseCommittedH\x00\x42\x05\n\x03msg\"\x1f\n\x1d\x43onversationResponseCommitted\"\xa5\x01\n\x16\x43onversationTurnPolicy\x12&\n\x1e\x61llow_interrupt_while_speaking\x18\x01 \x01(\x08\x12!\n\x19min_interrupt_duration_ms\x18\x02 \x01(\r\x12 \n\x18max_endpointing_delay_ms\x18\x03 \x01(\r\x12\x1e\n\x16stable_speaking_min_ms\x18\x04 \x01(\r\"\xa4\x01\n\x19\x43onversationSessionUpdate\x12\x11\n\tstt_model\x18\x01 \x01(\t\x12\x11\n\ttts_model\x18\x02 \x01(\t\x12\r\n\x05voice\x18\x03 \x01(\t\x12\x10\n\x08language\x18\x04 \x01(\t\x12\x13\n\x0bsample_rate\x18\x05 \x01(\r\x12+\n\x06policy\x18\x06 \x01(\x0b\x32\x1b.vox.ConversationTurnPolicy\"=\n\x17\x43onversationAudioAppend\x12\r\n\x05pcm16\x18\x01 \x01(\x0c\x12\x13\n\x0bsample_rate\x18\x02 \x01(\r\"\x1b\n\x19\x43onversationResponseStart\"*\n\x19\x43onversationResponseDelta\x12\r\n\x05\x64\x65lta\x18\x01 \x01(\t\"\x1c\n\x1a\x43onversationResponseCommit\"\x1c\n\x1a\x43onversationResponseCancel\"\x1c\n\x1a\x43onversationSessionCreated\"1\n\x19\x43onversationSpeechStarted\x12\x14\n\x0ctimestamp_ms\x18\x01 \x01(\r\"1\n\x19\x43onversationSpeechStopped\x12\x14\n\x0ctimestamp_ms\x18\x01 \x01(\r\"\xe8\x01\n\x1a\x43onversationTranscriptDone\x12\x12\n\ntranscript\x18\x01 \x01(\t\x12\x10\n\x08language\x18\x02 \x01(\t\x12\x10\n\x08start_ms\x18\x03 \x01(\r\x12\x0e\n\x06\x65nd_ms\x18\x04 \x01(\r\x12\x1c\n\x0f\x65ou_probability\x18\x05 \x01(\x02H\x00\x88\x01\x01\x12\x1d\n\x08\x65ntities\x18\x06 \x03(\x0b\x32\x0b.vox.Entity\x12\x0e\n\x06topics\x18\x07 \x03(\t\x12!\n\x05words\x18\x08 \x03(\x0b\x32\x12.vox.WordTimestampB\x12\n\x10_eou_probability\"\x1d\n\x1b\x43onversationResponseCreated\"<\n\x16\x43onversationAudioDelta\x12\r\n\x05\x61udio\x18\x01 \x01(\x0c\x12\x13\n\x0bsample_rate\x18\x02 \x01(\r\"\x1a\n\x18\x43onversationResponseDone\"\x1f\n\x1d\x43onversationResponseCancelled\"A\n\x18\x43onversationStateChanged\x12\r\n\x05state\x18\x01 \x01(\t\x12\x16\n\x0eprevious_state\x18\x02 \x01(\t\"$\n\x11\x43onversationError\x12\x0f\n\x07message\x18\x01 \x01(\t2\x81\x01\n\rHealthService\x12\x31\n\x06Health\x12\x12.vox.HealthRequest\x1a\x13.vox.HealthResponse\x12=\n\nListLoaded\x12\x16.vox.ListLoadedRequest\x1a\x17.vox.ListLoadedResponse2\xd6\x01\n\x0cModelService\x12-\n\x04Pull\x12\x10.vox.PullRequest\x1a\x11.vox.PullProgress0\x01\x12\x37\n\x04List\x12\x16.vox.ListModelsRequest\x1a\x17.vox.ListModelsResponse\x12+\n\x04Show\x12\x10.vox.ShowRequest\x1a\x11.vox.ShowResponse\x12\x31\n\x06\x44\x65lete\x12\x12.vox.DeleteRequest\x1a\x13.vox.DeleteResponse2\x8e\x01\n\x14TranscriptionService\x12=\n\nTranscribe\x12\x16.vox.TranscribeRequest\x1a\x17.vox.TranscribeResponse\x12\x37\n\x08\x41nnotate\x12\x14.vox.AnnotateRequest\x1a\x15.vox.AnnotateResponse2\x8e\x02\n\x10SynthesisService\x12\x37\n\nSynthesize\x12\x16.vox.SynthesizeRequest\x1a\x0f.vox.AudioChunk0\x01\x12=\n\nListVoices\x12\x16.vox.ListVoicesRequest\x1a\x17.vox.ListVoicesResponse\x12@\n\x0b\x43reateVoice\x12\x17.vox.CreateVoiceRequest\x1a\x18.vox.CreateVoiceResponse\x12@\n\x0b\x44\x65leteVoice\x12\x17.vox.DeleteVoiceRequest\x1a\x18.vox.DeleteVoiceResponse2O\n\x10StreamingService\x12;\n\x10StreamTranscribe\x12\x10.vox.StreamInput\x1a\x11.vox.StreamOutput(\x01\x30\x01\x32]\n\x13\x43onversationService\x12\x46\n\x08\x43onverse\x12\x1a.vox.ConverseClientMessage\x1a\x1a.vox.ConverseServerMessage(\x01\x30\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'vox_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_SHOWRESPONSE_CONFIGENTRY']._loaded_options = None
  _globals['_SHOWRESPONSE_CONFIGENTRY']._serialized_options = b'8\001'
  _globals['_HEALTHREQUEST']._serialized_start=18
  _globals['_HEALTHREQUEST']._serialized_end=33
  _globals['_HEALTHRESPONSE']._serialized_start=35
  _globals['_HEALTHRESPONSE']._serialized_end=67
  _globals['_LISTLOADEDREQUEST']._serialized_start=69
  _globals['_LISTLOADEDREQUEST']._serialized_end=88
  _globals['_LISTLOADEDRESPONSE']._serialized_start=90
  _globals['_LISTLOADEDRESPONSE']._serialized_end=144
  _globals['_LOADEDMODEL']._serialized_start=147
  _globals['_LOADEDMODEL']._serialized_end=294
  _globals['_PULLREQUEST']._serialized_start=296
  _globals['_PULLREQUEST']._serialized_end=323
  _globals['_PULLPROGRESS']._serialized_start=325
  _globals['_PULLPROGRESS']._serialized_end=404
  _globals['_LISTMODELSREQUEST']._serialized_start=406
  _globals['_LISTMODELSREQUEST']._serialized_end=425
  _globals['_LISTMODELSRESPONSE']._serialized_start=427
  _globals['_LISTMODELSRESPONSE']._serialized_end=479
  _globals['_MODELINFO']._serialized_start=481
  _globals['_MODELINFO']._serialized_end=599
  _globals['_SHOWREQUEST']._serialized_start=601
  _globals['_SHOWREQUEST']._serialized_end=628
  _globals['_SHOWRESPONSE']._serialized_start=631
  _globals['_SHOWRESPONSE']._serialized_end=785
  _globals['_SHOWRESPONSE_CONFIGENTRY']._serialized_start=740
  _globals['_SHOWRESPONSE_CONFIGENTRY']._serialized_end=785
  _globals['_LAYERINFO']._serialized_start=787
  _globals['_LAYERINFO']._serialized_end=866
  _globals['_DELETEREQUEST']._serialized_start=868
  _globals['_DELETEREQUEST']._serialized_end=897
  _globals['_DELETERESPONSE']._serialized_start=899
  _globals['_DELETERESPONSE']._serialized_end=931
  _globals['_TRANSCRIBEREQUEST']._serialized_start=934
  _globals['_TRANSCRIBEREQUEST']._serialized_end=1093
  _globals['_TRANSCRIBERESPONSE']._serialized_start=1096
  _globals['_TRANSCRIBERESPONSE']._serialized_end=1296
  _globals['_ENTITY']._serialized_start=1298
  _globals['_ENTITY']._serialized_end=1372
  _globals['_ANNOTATEREQUEST']._serialized_start=1374
  _globals['_ANNOTATEREQUEST']._serialized_end=1423
  _globals['_ANNOTATERESPONSE']._serialized_start=1425
  _globals['_ANNOTATERESPONSE']._serialized_end=1490
  _globals['_TRANSCRIPTSEGMENT']._serialized_start=1492
  _globals['_TRANSCRIPTSEGMENT']._serialized_end=1594
  _globals['_WORDTIMESTAMP']._serialized_start=1596
  _globals['_WORDTIMESTAMP']._serialized_end=1699
  _globals['_SYNTHESIZEREQUEST']._serialized_start=1701
  _globals['_SYNTHESIZEREQUEST']._serialized_end=1823
  _globals['_AUDIOCHUNK']._serialized_start=1825
  _globals['_AUDIOCHUNK']._serialized_end=1891
  _globals['_STREAMINPUT']._serialized_start=1894
  _globals['_STREAMINPUT']._serialized_end=2115
  _globals['_ENDOFSTREAM']._serialized_start=2117
  _globals['_ENDOFSTREAM']._serialized_end=2130
  _globals['_STREAMCONFIG']._serialized_start=2133
  _globals['_STREAMCONFIG']._serialized_end=2327
  _globals['_AUDIOFRAME']._serialized_start=2329
  _globals['_AUDIOFRAME']._serialized_end=2377
  _globals['_OPUSFRAME']._serialized_start=2379
  _globals['_OPUSFRAME']._serialized_end=2443
  _globals['_ENCODEDAUDIOFRAME']._serialized_start=2445
  _globals['_ENCODEDAUDIOFRAME']._serialized_end=2494
  _globals['_STREAMOUTPUT']._serialized_start=2497
  _globals['_STREAMOUTPUT']._serialized_end=2750
  _globals['_STREAMREADY']._serialized_start=2752
  _globals['_STREAMREADY']._serialized_end=2765
  _globals['_STREAMSPEECHSTARTED']._serialized_start=2767
  _globals['_STREAMSPEECHSTARTED']._serialized_end=2810
  _globals['_STREAMSPEECHSTOPPED']._serialized_start=2812
  _globals['_STREAMSPEECHSTOPPED']._serialized_end=2855
  _globals['_STREAMTRANSCRIPTRESULT']._serialized_start=2858
  _globals['_STREAMTRANSCRIPTRESULT']._serialized_end=3198
  _globals['_STREAMERRORMESSAGE']._serialized_start=3200
  _globals['_STREAMERRORMESSAGE']._serialized_end=3237
  _globals['_LISTVOICESREQUEST']._serialized_start=3239
  _globals['_LISTVOICESREQUEST']._serialized_end=3273
  _globals['_LISTVOICESRESPONSE']._serialized_start=3275
  _globals['_LISTVOICESRESPONSE']._serialized_end=3327
  _globals['_CREATEVOICEREQUEST']._serialized_start=3330
  _globals['_CREATEVOICEREQUEST']._serialized_end=3458
  _globals['_CREATEVOICERESPONSE']._serialized_start=3460
  _globals['_CREATEVOICERESPONSE']._serialized_end=3532
  _globals['_DELETEVOICEREQUEST']._serialized_start=3534
  _globals['_DELETEVOICEREQUEST']._serialized_end=3566
  _globals['_DELETEVOICERESPONSE']._serialized_start=3568
  _globals['_DELETEVOICERESPONSE']._serialized_end=3618
  _globals['_VOICEINFO']._serialized_start=3620
  _globals['_VOICEINFO']._serialized_end=3746
  _globals['_CONVERSECLIENTMESSAGE']._serialized_start=3749
  _globals['_CONVERSECLIENTMESSAGE']._serialized_end=4150
  _globals['_CONVERSESERVERMESSAGE']._serialized_start=4153
  _globals['_CONVERSESERVERMESSAGE']._serialized_end=4818
  _globals['_CONVERSATIONRESPONSECOMMITTED']._serialized_start=4820
  _globals['_CONVERSATIONRESPONSECOMMITTED']._serialized_end=4851
  _globals['_CONVERSATIONTURNPOLICY']._serialized_start=4854
  _globals['_CONVERSATIONTURNPOLICY']._serialized_end=5019
  _globals['_CONVERSATIONSESSIONUPDATE']._serialized_start=5022
  _globals['_CONVERSATIONSESSIONUPDATE']._serialized_end=5186
  _globals['_CONVERSATIONAUDIOAPPEND']._serialized_start=5188
  _globals['_CONVERSATIONAUDIOAPPEND']._serialized_end=5249
  _globals['_CONVERSATIONRESPONSESTART']._serialized_start=5251
  _globals['_CONVERSATIONRESPONSESTART']._serialized_end=5278
  _globals['_CONVERSATIONRESPONSEDELTA']._serialized_start=5280
  _globals['_CONVERSATIONRESPONSEDELTA']._serialized_end=5322
  _globals['_CONVERSATIONRESPONSECOMMIT']._serialized_start=5324
  _globals['_CONVERSATIONRESPONSECOMMIT']._serialized_end=5352
  _globals['_CONVERSATIONRESPONSECANCEL']._serialized_start=5354
  _globals['_CONVERSATIONRESPONSECANCEL']._serialized_end=5382
  _globals['_CONVERSATIONSESSIONCREATED']._serialized_start=5384
  _globals['_CONVERSATIONSESSIONCREATED']._serialized_end=5412
  _globals['_CONVERSATIONSPEECHSTARTED']._serialized_start=5414
  _globals['_CONVERSATIONSPEECHSTARTED']._serialized_end=5463
  _globals['_CONVERSATIONSPEECHSTOPPED']._serialized_start=5465
  _globals['_CONVERSATIONSPEECHSTOPPED']._serialized_end=5514
  _globals['_CONVERSATIONTRANSCRIPTDONE']._serialized_start=5517
  _globals['_CONVERSATIONTRANSCRIPTDONE']._serialized_end=5749
  _globals['_CONVERSATIONRESPONSECREATED']._serialized_start=5751
  _globals['_CONVERSATIONRESPONSECREATED']._serialized_end=5780
  _globals['_CONVERSATIONAUDIODELTA']._serialized_start=5782
  _globals['_CONVERSATIONAUDIODELTA']._serialized_end=5842
  _globals['_CONVERSATIONRESPONSEDONE']._serialized_start=5844
  _globals['_CONVERSATIONRESPONSEDONE']._serialized_end=5870
  _globals['_CONVERSATIONRESPONSECANCELLED']._serialized_start=5872
  _globals['_CONVERSATIONRESPONSECANCELLED']._serialized_end=5903
  _globals['_CONVERSATIONSTATECHANGED']._serialized_start=5905
  _globals['_CONVERSATIONSTATECHANGED']._serialized_end=5970
  _globals['_CONVERSATIONERROR']._serialized_start=5972
  _globals['_CONVERSATIONERROR']._serialized_end=6008
  _globals['_HEALTHSERVICE']._serialized_start=6011
  _globals['_HEALTHSERVICE']._serialized_end=6140
  _globals['_MODELSERVICE']._serialized_start=6143
  _globals['_MODELSERVICE']._serialized_end=6357
  _globals['_TRANSCRIPTIONSERVICE']._serialized_start=6360
  _globals['_TRANSCRIPTIONSERVICE']._serialized_end=6502
  _globals['_SYNTHESISSERVICE']._serialized_start=6505
  _globals['_SYNTHESISSERVICE']._serialized_end=6775
  _globals['_STREAMINGSERVICE']._serialized_start=6777
  _globals['_STREAMINGSERVICE']._serialized_end=6856
  _globals['_CONVERSATIONSERVICE']._serialized_start=6858
  _globals['_CONVERSATIONSERVICE']._serialized_end=6951

